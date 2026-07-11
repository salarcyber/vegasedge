"""MODULE 2 — Feature engineering & preprocessing.

Builds one model-ready row per game: matchup differentials of advanced metrics
(Net Rating, EPA/play, xG, wRC+), plus Module 7 context (rest, travel, weather,
sentiment, situational splits) and the market's own opening line (the single
strongest feature — beating the market means modelling *deviation* from it).

Missing-data policy:
  * numeric features -> median impute (fitted on train only)
  * missing deep-data signals -> impute neutral 0 (a differential of 0 = no edge)
  * everything scaled with StandardScaler inside a sklearn Pipeline so the exact
    same transform is applied at train and inference time.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.db import query

# Which jsonb metrics feed the model, per sport. Differential = home - away.
SPORT_METRICS = {
    # game-derived scoring metrics (robust everywhere); nba_api advanced metrics
    # (net_rating etc.) get merged in by the live pipeline when reachable
    "nba": ["pf_pg", "pa_pg", "net_pg", "win_pct"],
    "nfl": ["epa_off", "epa_def", "success_off", "success_def"],
    "mlb": ["ops", "runs_pg", "era", "whip"],
    "nhl": ["gf_per_game", "ga_per_game", "point_pct"],
    "soccer_epl": ["xg_for", "xg_against", "ppda"],
}

# per-sport extra features beyond team metrics + shared context
SPORT_EXTRAS = {
    "mlb": ["d_sp_era", "d_sp_whip", "d_sp_k9"],   # starting pitcher matchup
}

# point-in-time form, computed from our own game history at each game's date
FORM_FEATURES = ["d_form_win15", "d_form_net15"]

# Deep-data features are capped: sentiment/travel are weak, noisy signals and
# must never dominate the core efficiency metrics. XGBoost learns its own
# weights, but monotone constraints + these caps encode sane priors.
CONTEXT_FEATURES = [
    "rest_diff",            # home rest days - away rest days
    "travel_diff_km",       # away travel - home travel (positive favours home)
    "tz_crossed_away",
    "sentiment_diff",       # home sentiment - away sentiment, clipped to [-1, 1]
    "wind_kph",             # outdoor sports: totals killer
    "precip_mm",
    "b2b_away",             # away team on a back-to-back (situational split flag)
    "market_home_prob_open",  # devigged opening line — the market prior
]


def build_matchup_frame(conn, sport: str, include_labels: bool) -> pd.DataFrame:
    """One row per game with differential features. include_labels=True pulls
    finished games (training); False pulls upcoming games (inference)."""
    status = "= 'final'" if include_labels else "= 'scheduled'"
    form_sql = """
        select avg((s.pf > s.pa)::int) win15, avg(s.pf - s.pa) net15, count(*) n
        from (
            select case when g2.home_team_id = %(tid)s then g2.home_score
                        else g2.away_score end pf,
                   case when g2.home_team_id = %(tid)s then g2.away_score
                        else g2.home_score end pa
            from games g2
            where (g2.home_team_id = %(tid)s or g2.away_team_id = %(tid)s)
              and g2.status = 'final' and g2.commence_time < g.commence_time
            order by g2.commence_time desc limit 15
        ) s
    """
    games = query(conn, f"""
        select g.event_id, g.commence_time, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.home_rest_days, g.away_rest_days,
               g.home_travel_km, g.away_travel_km,
               hm.metrics home_m, am.metrics away_m,
               gc.weather, gc.sentiment, gc.tz_crossed_away, gc.notes,
               hf.win15 h_win15, hf.net15 h_net15, hf.n h_form_n,
               af.win15 a_win15, af.net15 a_net15, af.n a_form_n
        from games g
        left join lateral (select metrics from team_metrics
            where team_id = g.home_team_id and as_of <= g.commence_time::date
            order by as_of desc limit 1) hm on true
        left join lateral (select metrics from team_metrics
            where team_id = g.away_team_id and as_of <= g.commence_time::date
            order by as_of desc limit 1) am on true
        left join lateral ({form_sql.replace('%(tid)s', 'g.home_team_id')}) hf on true
        left join lateral ({form_sql.replace('%(tid)s', 'g.away_team_id')}) af on true
        left join game_context gc on gc.event_id = g.event_id
        where g.sport = %s and g.status {status}
    """, (sport,))

    rows = []
    for g in games:
        hm, am = g["home_m"] or {}, g["away_m"] or {}
        row: dict = {"event_id": g["event_id"]}
        for m in SPORT_METRICS.get(sport, []):
            h, a = hm.get(m), am.get(m)
            row[f"d_{m}"] = (h - a) if (h is not None and a is not None) else None
        row["rest_diff"] = (
            (g["home_rest_days"] - g["away_rest_days"])
            if g["home_rest_days"] is not None and g["away_rest_days"] is not None else None)
        row["travel_diff_km"] = (
            (g["away_travel_km"] or 0) - (g["home_travel_km"] or 0)
            if g["away_travel_km"] is not None or g["home_travel_km"] is not None else None)
        row["tz_crossed_away"] = g["tz_crossed_away"]
        sent = g["sentiment"] or {}
        hs = (sent.get("home") or {}).get("score")
        as_ = (sent.get("away") or {}).get("score")
        row["sentiment_diff"] = max(-1, min(1, hs - as_)) if hs is not None and as_ is not None else None
        w = g["weather"] or {}
        row["wind_kph"] = w.get("wind_kph")
        row["precip_mm"] = w.get("precip_mm")
        row["b2b_away"] = 1 if (g["away_rest_days"] is not None and g["away_rest_days"] <= 1) else 0
        row["market_home_prob_open"] = _opening_market_prob(conn, g["event_id"], g["home_team_id"])
        # point-in-time form (require a real sample so early-season noise imputes)
        if (g["h_form_n"] or 0) >= 5 and (g["a_form_n"] or 0) >= 5:
            row["d_form_win15"] = round(float(g["h_win15"] - g["a_win15"]), 4)
            row["d_form_net15"] = round(float(g["h_net15"] - g["a_net15"]), 4)
        else:
            row["d_form_win15"] = row["d_form_net15"] = None
        # starting pitcher matchup (MLB)
        notes = g.get("notes") or {}
        hsp, asp = notes.get("home_sp") or {}, notes.get("away_sp") or {}
        for stat in ("era", "whip", "k9"):
            h, a = hsp.get(stat), asp.get(stat)
            row[f"d_sp_{stat}"] = round(h - a, 3) if (h is not None and a is not None) else None
        if include_labels and g["home_score"] is not None:
            row["home_win"] = int(g["home_score"] > g["away_score"])
            row["total_points"] = g["home_score"] + g["away_score"]
            row["home_margin"] = g["home_score"] - g["away_score"]
        rows.append(row)
    return pd.DataFrame(rows)


def _opening_market_prob(conn, event_id: str, home_team_id: str) -> float | None:
    """Market prior for the model: devigged PINNACLE (sharpest book) when we
    have it, else the devigged opening consensus."""
    from src.utils.odds_math import devig_multiplicative

    snaps = query(conn, """
        select distinct on (outcome) outcome, price_decimal from odds_snapshots
        where event_id = %s and market = 'h2h' and bookmaker = 'pinnacle'
        order by outcome, captured_at desc
    """, (event_id,))
    if len(snaps) < 2:
        snaps = query(conn, """
            select outcome, price_decimal from odds_snapshots
            where event_id = %s and market = 'h2h' and is_opening
        """, (event_id,))
    if len(snaps) < 2:
        return None
    decs = [s["price_decimal"] for s in snaps]
    probs = devig_multiplicative(decs)
    for s, p in zip(snaps, probs):
        # suffix match instead of split("_", 1): sport keys contain underscores
        # (soccer_epl_Arsenal), so a prefix split never matches soccer outcomes
        if home_team_id.endswith("_" + s["outcome"]):
            return round(p, 4)
    return None


def feature_columns(sport: str) -> list[str]:
    return ([f"d_{m}" for m in SPORT_METRICS.get(sport, [])]
            + SPORT_EXTRAS.get(sport, []) + FORM_FEATURES + CONTEXT_FEATURES)


def make_preprocessor(sport: str) -> ColumnTransformer:
    cols = feature_columns(sport)
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("num", num, cols)], remainder="drop")

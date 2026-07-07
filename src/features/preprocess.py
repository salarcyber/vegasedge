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
    "nba": ["net_rating", "off_rating", "def_rating", "pace", "efg_pct", "tov_pct"],
    "nfl": ["epa_off", "epa_def", "success_off", "success_def"],
    "mlb": ["ops", "runs_pg", "era", "whip"],
    "nhl": ["gf_per_game", "ga_per_game", "point_pct"],
    "soccer_epl": ["xg_for", "xg_against", "ppda"],
}

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
    games = query(conn, f"""
        select g.event_id, g.commence_time, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.home_rest_days, g.away_rest_days,
               g.home_travel_km, g.away_travel_km,
               hm.metrics home_m, am.metrics away_m,
               gc.weather, gc.sentiment, gc.tz_crossed_away
        from games g
        left join lateral (select metrics from team_metrics
            where team_id = g.home_team_id and as_of <= g.commence_time::date
            order by as_of desc limit 1) hm on true
        left join lateral (select metrics from team_metrics
            where team_id = g.away_team_id and as_of <= g.commence_time::date
            order by as_of desc limit 1) am on true
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
        if include_labels and g["home_score"] is not None:
            row["home_win"] = int(g["home_score"] > g["away_score"])
            row["total_points"] = g["home_score"] + g["away_score"]
            row["home_margin"] = g["home_score"] - g["away_score"]
        rows.append(row)
    return pd.DataFrame(rows)


def _opening_market_prob(conn, event_id: str, home_team_id: str) -> float | None:
    from src.utils.odds_math import devig_multiplicative

    snaps = query(conn, """
        select outcome, price_decimal from odds_snapshots
        where event_id = %s and market = 'h2h' and is_opening
    """, (event_id,))
    if len(snaps) < 2:
        return None
    home_name = home_team_id.split("_", 1)[1]
    decs = [s["price_decimal"] for s in snaps]
    probs = devig_multiplicative(decs)
    for s, p in zip(snaps, probs):
        if s["outcome"] == home_name:
            return round(p, 4)
    return None


def feature_columns(sport: str) -> list[str]:
    return [f"d_{m}" for m in SPORT_METRICS.get(sport, [])] + CONTEXT_FEATURES


def make_preprocessor(sport: str) -> ColumnTransformer:
    cols = feature_columns(sport)
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("num", num, cols)], remainder="drop")

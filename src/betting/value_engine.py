"""MODULE 3 — Value Bet Engine.

Pipeline per upcoming game:
  1. Get calibrated true probability from Module 2 (or Poisson for soccer/NHL).
  2. Pull the freshest odds snapshot across all books; find the best price.
  3. Devig the market consensus, compute EV against the BEST available price.
  4. Flag value only when EV > MIN_EV, size with quarter-Kelly, store in
     `predictions` for the LLM analyst (Module 5) and dashboard (Module 8).

Run: python -m src.betting.value_engine [nba nfl ...]
"""
from __future__ import annotations

import sys
from datetime import date

from src.features.preprocess import build_matchup_frame
from src.models.predictor import predict_true_prob
from src.utils.db import get_conn, insert_many, query
from src.utils.odds_math import evaluate_bet, expected_lambdas, match_probs

MIN_EV = 0.02          # demand 2%+ edge; anything less is model noise
MAX_EV = 0.20          # edges above 20% are almost always model error, not
                       # market error — flag them as no-bet until proven
MAX_EV_SOCCER = 0.08   # backtest 2023-26 (1,846 bets vs closing): 2-8% edges
                       # +1.05% flat ROI, 8-20% "edges" -1.64% — cap them out
KELLY_MULT = 0.25      # quarter-Kelly
DEFAULT_BANKROLL = 1000.0


def current_bankroll(conn) -> float:
    r = query(conn, "select balance from bankroll order by ts desc limit 1")
    return r[0]["balance"] if r else DEFAULT_BANKROLL


def prob_shrink(conn, sport: str) -> float:
    r = query(conn, """select prob_shrink from model_calibration
                       where sport = %s order by window_end desc limit 1""", (sport,))
    return float(r[0]["prob_shrink"]) if r else 0.5


def latest_market(conn, event_id: str, market: str = "h2h") -> dict[str, dict]:
    """Freshest price per outcome per book -> {outcome: {best_decimal, best_book, all_decimals}}"""
    snaps = query(conn, """
        select distinct on (bookmaker, outcome) bookmaker, outcome, price_decimal
        from odds_snapshots
        where event_id = %s and market = %s
          -- drop books that stopped quoting: a delisted book's frozen price
          -- stays "freshest" for that book forever, and because the market
          -- moved away from it, it wins best-price selection and inflates EV
          and captured_at >= (
            select max(captured_at) - interval '2 hours'
            from odds_snapshots where event_id = %s and market = %s)
        order by bookmaker, outcome, captured_at desc
    """, (event_id, market, event_id, market))
    by_outcome: dict[str, dict] = {}
    for s in snaps:
        o = by_outcome.setdefault(s["outcome"], {"best_decimal": 0, "best_book": None, "decimals": []})
        o["decimals"].append(s["price_decimal"])
        if s["price_decimal"] > o["best_decimal"]:
            o["best_decimal"], o["best_book"] = s["price_decimal"], s["bookmaker"]
    return by_outcome


def moneyline_candidates(conn, sport: str, bankroll: float) -> list[dict]:
    df = build_matchup_frame(conn, sport, include_labels=False)
    if df.empty:
        return []
    preds = predict_true_prob(df, sport, prob_shrink(conn, sport))
    out = []
    for _, p in preds.iterrows():
        game = query(conn, "select home_team_id, away_team_id from games where event_id=%s",
                     (p["event_id"],))[0]
        # strip the exact sport prefix — split("_", 1) breaks on sports whose
        # key contains underscores (soccer_epl_Arsenal -> "epl_Arsenal")
        home = game["home_team_id"][len(sport) + 1:]
        away = game["away_team_id"][len(sport) + 1:]
        market = latest_market(conn, p["event_id"])
        for outcome, true_p in ((home, p["model_prob"]), (away, 1 - p["model_prob"])):
            m = market.get(outcome)
            if not m:
                continue
            all_decs = [market[o]["best_decimal"] for o in market]
            ev = evaluate_bet(true_p, m["best_decimal"], all_decs, bankroll,
                              MIN_EV, KELLY_MULT)
            out.append({
                "event_id": p["event_id"], "market": "h2h", "outcome": outcome,
                "line": None, "model_prob": ev.true_prob,
                "fair_decimal": round(1 / max(ev.true_prob, 1e-6), 3),
                "best_book": m["best_book"], "best_decimal": m["best_decimal"],
                "ev_pct": ev.ev_pct, "kelly_frac": ev.kelly_frac,
                "model_version": p["model_version"], "is_value_bet": ev.is_value,
            })
    return out


def poisson_candidates(conn, sport: str, bankroll: float) -> list[dict]:
    """Soccer/NHL: derive 1X2 + totals probabilities from team xG via Poisson."""
    league_avg = 6.0 if sport == "nhl" else 2.8
    games = query(conn, """
        select g.event_id, g.home_team_id, g.away_team_id, hm.metrics hm, am.metrics am
        from games g
        left join lateral (select metrics from team_metrics where team_id=g.home_team_id
            order by as_of desc limit 1) hm on true
        left join lateral (select metrics from team_metrics where team_id=g.away_team_id
            order by as_of desc limit 1) am on true
        where g.sport = %s and g.status = 'scheduled'
    """, (sport,))
    out = []
    for g in games:
        hm, am = g["hm"] or {}, g["am"] or {}
        keys = ("xg_for", "xg_against") if "xg_for" in hm else ("gf_per_game", "ga_per_game")
        if not all(k in hm and k in am for k in keys):
            continue
        lam_h, lam_a = expected_lambdas(hm[keys[0]], hm[keys[1]],
                                        am[keys[0]], am[keys[1]], league_avg)
        probs = match_probs(lam_h, lam_a)
        market = latest_market(conn, g["event_id"])
        # strip the exact sport prefix: split("_", 1) yields "epl_Arsenal" for
        # soccer sports, which never matches odds outcomes — every home/away
        # soccer bet would be silently skipped, leaving Draw-only pricing
        name_map = {
            g["home_team_id"][len(sport) + 1:]: probs["home"],
            g["away_team_id"][len(sport) + 1:]: probs["away"],
            "Draw": probs["draw"],
        }
        all_decs = [m["best_decimal"] for m in market.values()]
        for outcome, true_p in name_map.items():
            m = market.get(outcome)
            if not m:
                continue
            ev = evaluate_bet(true_p, m["best_decimal"], all_decs, bankroll,
                              MIN_EV, KELLY_MULT, max_ev=MAX_EV_SOCCER)
            out.append({
                "event_id": g["event_id"], "market": "h2h", "outcome": outcome,
                "line": None, "model_prob": ev.true_prob,
                "fair_decimal": round(1 / max(ev.true_prob, 1e-6), 3),
                "best_book": m["best_book"], "best_decimal": m["best_decimal"],
                "ev_pct": ev.ev_pct, "kelly_frac": ev.kelly_frac,
                "model_version": f"poisson-{sport}", "is_value_bet": ev.is_value,
            })
    return out


def main(sports: list[str]) -> None:
    from src.ingestion.odds_ingest import DB_SPORT

    with get_conn() as conn:
        bankroll = current_bankroll(conn)
        all_rows: list[dict] = []
        for sport in sports:
            sport = DB_SPORT.get(sport, sport)
            try:
                if sport.startswith(("soccer", "worldcup", "nhl")):
                    rows = poisson_candidates(conn, sport, bankroll)
                else:
                    rows = moneyline_candidates(conn, sport, bankroll)
                all_rows.extend(rows)
                n_val = sum(r["is_value_bet"] for r in rows)
                print(f"[value] {sport}: {len(rows)} priced, {n_val} value bets")
            except FileNotFoundError:
                print(f"[value] {sport}: no trained model yet — run src.models.train first")
            except Exception as e:
                print(f"[value] {sport} failed: {e}")
        # replace, don't append: clear prior prices for the games being repriced
        # (keeps one live row per event/market/outcome; settled games untouched).
        # Carry the AI briefings over so repricing never blanks the dashboard
        # and the LLM writes each game up once, not every hour.
        event_ids = list({r["event_id"] for r in all_rows})
        if event_ids:
            old = query(conn, """select event_id, market, outcome, reasoning
                                 from predictions where event_id = any(%s)
                                   and reasoning is not null""", (event_ids,))
            kept = {(o["event_id"], o["market"], o["outcome"]): o["reasoning"] for o in old}
            for r in all_rows:
                r["reasoning"] = kept.get((r["event_id"], r["market"], r["outcome"]))
            with conn.cursor() as cur:
                cur.execute("""delete from predictions p using games g
                               where p.event_id = g.event_id and g.status = 'scheduled'
                                 and p.event_id = any(%s)""", (event_ids,))
        insert_many(conn, "predictions", all_rows)
        alert_new_picks(conn, [r for r in all_rows if r["is_value_bet"]], bankroll)


def alert_new_picks(conn, picks: list[dict], bankroll: float) -> None:
    """Push MODEL PICK alerts to Discord (free webhook) so value doesn't expire
    unseen. Set DISCORD_WEBHOOK in .env / GitHub secrets to enable."""
    import os

    hook = os.environ.get("DISCORD_WEBHOOK", "").strip()
    if not hook or not picks:
        return
    import httpx

    lines = []
    for p in picks[:8]:
        g = query(conn, """select home_team_id, away_team_id from games
                           where event_id = %s""", (p["event_id"],))
        if not g:
            continue
        home = g[0]["home_team_id"].split("_")[-1]
        away = g[0]["away_team_id"].split("_")[-1]
        stake = bankroll * (p["kelly_frac"] or 0)
        lines.append(f"🔥 **{p['outcome']}** ({away} @ {home}) — edge +{p['ev_pct']}%, "
                     f"bet ${stake:,.0f}")
    if not lines:
        return
    try:
        httpx.post(hook, json={"content": "**VegasEdge — new model picks**\n" +
                                          "\n".join(lines)}, timeout=15)
        print(f"[alert] sent {len(lines)} picks to Discord")
    except Exception as e:
        print(f"[alert] webhook failed: {e}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nba", "nfl", "epl"])

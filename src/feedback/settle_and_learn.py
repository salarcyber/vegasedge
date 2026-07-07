"""MODULE 6 — Retaining the Edge: automated feedback loop.

Runs daily (GitHub Actions):
  1. SETTLE   — pull final scores from ESPN's free scoreboard JSON, mark games
     final, grade every flagged bet, log P/L and update the bankroll curve.
  2. CLV      — record each bet's Closing Line Value: the price we took vs. the
     last snapshot before kickoff. CLV is the fastest honest signal of edge —
     if you consistently beat the close, profit follows; if you don't, no
     amount of short-term wins means the model works.
  3. ADAPT    — recompute rolling 90-day Brier/ROI and update `prob_shrink`,
     the weight the live engine puts on our model vs. the market prior:
        beat the market's calibration  -> shrink toward model (more aggressive)
        lose to the market             -> shrink toward market (defensive)
     This is the self-correcting governor that keeps the system from bleeding
     when the market adapts.
  4. RETRAIN  — models are retrained weekly by CI on the grown dataset, so new
     game outcomes continuously reshape feature weights.
"""
from __future__ import annotations

from datetime import date, timedelta

import httpx

from src.utils.db import get_conn, insert_many, query, upsert

ESPN_SCOREBOARD = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
    "soccer_epl": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
}


# ------------------------------------------------------------------ 1. settle

def fetch_finals(sport: str, day: date) -> list[dict]:
    r = httpx.get(ESPN_SCOREBOARD[sport], params={"dates": day.strftime("%Y%m%d")},
                  timeout=30)
    r.raise_for_status()
    out = []
    for ev in r.json().get("events", []):
        comp = ev["competitions"][0]
        if comp["status"]["type"]["completed"]:
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            out.append({"home_name": home["team"]["displayName"],
                        "away_name": away["team"]["displayName"],
                        "home_score": int(home["score"]), "away_score": int(away["score"])})
    return out


def settle_games(conn, sport: str) -> int:
    n = 0
    for d in (date.today() - timedelta(days=1), date.today()):
        try:
            finals = fetch_finals(sport, d)
        except Exception as e:
            print(f"[settle] {sport} {d} scoreboard failed: {e}")
            continue
        for f in finals:
            with conn.cursor() as cur:
                cur.execute("""
                    update games g set status='final', home_score=%s, away_score=%s
                    from teams ht, teams at
                    where ht.team_id = g.home_team_id and at.team_id = g.away_team_id
                      and ht.name = %s and at.name = %s and g.status != 'final'
                      and g.commence_time::date between %s and %s
                """, (f["home_score"], f["away_score"], f["home_name"], f["away_name"],
                      d - timedelta(days=1), d))
                n += cur.rowcount
    print(f"[settle] {sport}: {n} games marked final")
    return n


def grade_bets(conn) -> None:
    """Grade every unsettled flagged bet on a now-final game, with CLV."""
    open_bets = query(conn, """
        select p.*, g.home_team_id, g.away_team_id, g.home_score, g.away_score,
               g.commence_time
        from predictions p join games g using (event_id)
        left join bet_results br on br.pred_id = p.pred_id
        where p.is_value_bet and g.status = 'final' and br.pred_id is null
    """)
    bank = query(conn, "select balance from bankroll order by ts desc limit 1")
    balance = bank[0]["balance"] if bank else 1000.0

    results = []
    for b in open_bets:
        home = b["home_team_id"].split("_", 1)[1]
        winner = home if b["home_score"] > b["away_score"] else \
            b["away_team_id"].split("_", 1)[1] if b["away_score"] > b["home_score"] else "Draw"
        stake = round(balance * (b["kelly_frac"] or 0), 2)
        if b["market"] != "h2h":
            continue  # extend grading for spreads/totals/props as you add them
        won = b["outcome"] == winner
        pnl = round(stake * (b["best_decimal"] - 1), 2) if won else -stake

        close = query(conn, """
            select price_decimal from odds_snapshots
            where event_id=%s and market=%s and outcome=%s and captured_at < %s
            order by captured_at desc limit 1
        """, (b["event_id"], b["market"], b["outcome"], b["commence_time"]))
        closing = close[0]["price_decimal"] if close else None
        clv = round((b["best_decimal"] / closing - 1) * 100, 2) if closing else None

        results.append({"pred_id": b["pred_id"], "stake": stake,
                        "result": "win" if won else "loss", "pnl": pnl,
                        "closing_decimal": closing, "clv_pct": clv})
        balance += pnl

    insert_many(conn, "bet_results", results)
    if results:
        insert_many(conn, "bankroll", [{"balance": round(balance, 2)}])
    wins = sum(r["result"] == "win" for r in results)
    pnl = sum(r["pnl"] for r in results)
    print(f"[grade] {len(results)} bets settled: {wins}W-{len(results)-wins}L, "
          f"P/L {pnl:+.2f}, new bankroll {balance:.2f}")


# ------------------------------------------------------------------ 3. adapt

def recalibrate(conn, sport: str, window_days: int = 90) -> None:
    """Rolling scorecard -> new prob_shrink. The governor:
       * model Brier beats market Brier by >2%  -> trust model more (+0.05)
       * model worse than market               -> trust market more (-0.10, asymmetric:
         we de-risk faster than we re-risk)
       * positive CLV is required to increase aggression at all.
    """
    perf = query(conn, """
        select count(*) n,
               avg((br.result = 'win')::int) hit_rate,
               sum(br.pnl) / nullif(sum(br.stake), 0) roi,
               avg(br.clv_pct) avg_clv,
               avg(power(p.model_prob - (br.result='win')::int, 2)) model_brier,
               avg(power(1.0/p.best_decimal - (br.result='win')::int, 2)) market_brier
        from bet_results br join predictions p using (pred_id)
        join games g using (event_id)
        where g.sport = %s and br.settled_at > now() - make_interval(days => %s)
          and br.result in ('win','loss')
    """, (sport, window_days))[0]

    if not perf["n"] or perf["n"] < 20:
        print(f"[adapt] {sport}: only {perf['n'] or 0} settled bets — keeping defaults")
        return

    prev = query(conn, """select prob_shrink from model_calibration
                          where sport=%s order by window_end desc limit 1""", (sport,))
    shrink = float(prev[0]["prob_shrink"]) if prev else 0.5

    model_edge = (perf["market_brier"] or 0) - (perf["model_brier"] or 1)
    if model_edge > 0.002 and (perf["avg_clv"] or 0) > 0:
        shrink = min(shrink + 0.05, 0.85)
    elif model_edge < 0:
        shrink = max(shrink - 0.10, 0.15)

    ver = query(conn, """select model_version from predictions p join games g using(event_id)
                         where g.sport=%s order by p.created_at desc limit 1""", (sport,))
    upsert(conn, "model_calibration", [{
        "model_version": ver[0]["model_version"] if ver else "unknown",
        "sport": sport, "window_end": date.today(),
        "n_bets": perf["n"], "hit_rate": round(float(perf["hit_rate"]), 4),
        "roi": round(float(perf["roi"] or 0), 4),
        "brier": round(float(perf["model_brier"]), 4),
        "logloss": None, "prob_shrink": round(shrink, 2),
    }], ["model_version", "sport", "window_end"])
    print(f"[adapt] {sport}: n={perf['n']} roi={perf['roi']:+.1%} "
          f"clv={perf['avg_clv'] or 0:+.2f}% -> prob_shrink={shrink}")


if __name__ == "__main__":
    with get_conn() as conn:
        for sport in ESPN_SCOREBOARD:
            settle_games(conn, sport)
        grade_bets(conn)
        for sport in ESPN_SCOREBOARD:
            recalibrate(conn, sport)
        with conn.cursor() as cur:
            cur.execute("select prune_old_snapshots()")

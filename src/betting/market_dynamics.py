"""MODULE 4 — Market anomaly detection & tournament logic switches.

Part A — Sharp vs. public divergence
------------------------------------
Two free signals, both derived from data we already store:

  1. REVERSE LINE MOVEMENT (RLM): line moves AGAINST the side receiving the
     majority of public tickets. Public on Home 70%, yet Home -7 drops to -6
     => books are respecting sharp money on Away. Requires public_betting rows
     (scrape from a consensus page) — but even without them:
  2. STEAM / OPENER DRIFT: pure line-history signal from odds_snapshots.
     A fast, synchronized move across many books = sharp origin. A slow drift
     at one book = position balancing. We score each event on both.

Part B — Tournament switches (World Cup / knockout formats)
-----------------------------------------------------------
Deterministic adjustments applied to model probabilities BEFORE the EV gate.
Each is a small multiplicative nudge with a hard cap — context should tilt a
close call, never overrule the core model.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.db import get_conn, query

# ------------------------------------------------------------------ Part A

STEAM_THRESHOLD_PCT = 3.0    # implied-prob move (pts) within one hour = steam
RLM_MIN_PUBLIC = 0.60        # public must be >=60% on the side for RLM to count


def line_movement_report(conn, event_id: str, market: str = "h2h") -> list[dict]:
    """Opening vs current implied probability per outcome, plus max 1-hour swing."""
    return query(conn, """
        with series as (
            select outcome, captured_at, avg(1.0 / price_decimal) as prob
            from odds_snapshots
            where event_id = %s and market = %s
            group by outcome, captured_at
        ),
        hourly as (
            select outcome,
                   max(abs(prob - lag(prob) over (partition by outcome order by captured_at))) as max_step
            from series group by outcome, prob, captured_at
        )
        select s.outcome,
               (array_agg(s.prob order by s.captured_at asc))[1]  as open_prob,
               (array_agg(s.prob order by s.captured_at desc))[1] as curr_prob,
               max(h.max_step) as max_hourly_step
        from series s join hourly h using (outcome)
        group by s.outcome
    """, (event_id, market))


@dataclass
class MarketSignal:
    outcome: str
    steam: bool                 # fast synchronized sharp move toward this side
    rlm: bool                   # line moved here against public money
    drift_pts: float            # total implied prob move since open (points)


def detect_anomalies(conn, event_id: str) -> list[MarketSignal]:
    moves = line_movement_report(conn, event_id)
    public = {p["side"]: p for p in query(conn, """
        select distinct on (side) side, bet_pct, money_pct from public_betting
        where event_id = %s order by side, captured_at desc""", (event_id,))}

    signals = []
    for m in moves:
        if m["open_prob"] is None or m["curr_prob"] is None:
            continue
        drift = (m["curr_prob"] - m["open_prob"]) * 100
        steam = (m["max_hourly_step"] or 0) * 100 >= STEAM_THRESHOLD_PCT and drift > 0
        # RLM: this side's line IMPROVED for bettors (drift toward it) while the
        # public is loaded on the OTHER side.
        rlm = False
        for side, p in public.items():
            if side != m["outcome"] and (p["bet_pct"] or 0) >= RLM_MIN_PUBLIC * 100 and drift > 1.0:
                rlm = True
            # money% >> bet% on this side = few large (sharp) wagers
            if side == m["outcome"] and p["money_pct"] and p["bet_pct"] \
               and p["money_pct"] - p["bet_pct"] >= 15:
                rlm = True
        signals.append(MarketSignal(m["outcome"], steam, rlm, round(drift, 2)))
    return signals


def anomaly_boost(signal: MarketSignal) -> float:
    """Probability multiplier for the EV engine. Sharp confirmation earns a
    small boost; never more than 4% total — the market signal supplements the
    model, it doesn't replace it."""
    boost = 1.0
    if signal.steam:
        boost *= 1.02
    if signal.rlm:
        boost *= 1.02
    return min(boost, 1.04)


# ------------------------------------------------------------------ Part B

@dataclass
class TournamentContext:
    stage: str = "group"            # group | knockout | final
    needs_draw_only: bool = False   # team advances with a draw (group stage)
    dead_rubber: bool = False       # already eliminated / already qualified
    rest_days: int = 4
    travel_km: float = 0.0
    is_neutral_venue: bool = True


def tournament_adjust(base_probs: dict[str, float], home_ctx: TournamentContext,
                      away_ctx: TournamentContext) -> dict[str, float]:
    """Apply knockout/motivation/fatigue switches to {home, draw, away} probs,
    then renormalize. Magnitudes are priors from World Cup research; tune with
    Module 6 once you have settled tournament bets."""
    p = dict(base_probs)

    # 1. Knockout stage: 90-min draw goes to extra time -> for "to advance"
    #    markets convert draw mass ~55/45 by penalty/ET strength proxy (favourite
    #    edges it slightly). For 1X2 (90 min) markets leave draws intact but
    #    knockout matches are cagier: shift 3% from both win probs into draw.
    if home_ctx.stage in ("knockout", "final"):
        shift = 0.03
        p["home"] -= p["home"] * shift
        p["away"] -= p["away"] * shift
        p["draw"] = 1 - p["home"] - p["away"]

    # 2. Motivation: a side that only needs a draw plays for it — draw prob up,
    #    their win prob down.
    for side in ("home", "away"):
        ctx = home_ctx if side == "home" else away_ctx
        if ctx.needs_draw_only:
            take = p[side] * 0.10
            p[side] -= take
            p["draw"] += take
        if ctx.dead_rubber:                       # nothing to play for
            other = "away" if side == "home" else "home"
            take = p[side] * 0.08
            p[side] -= take
            p[other] += take * 0.6
            p["draw"] += take * 0.4

    # 3. Fatigue: short rest (<4 days in tournament play) and long travel
    for side in ("home", "away"):
        ctx = home_ctx if side == "home" else away_ctx
        penalty = 0.0
        if ctx.rest_days <= 3:
            penalty += 0.03 * (4 - ctx.rest_days)   # 3 days rest = -3%, 2 = -6%
        if ctx.travel_km > 1500:
            penalty += 0.02
        if penalty:
            take = p[side] * min(penalty, 0.10)
            p[side] -= take
            other = "away" if side == "home" else "home"
            p[other] += take * 0.7
            p["draw"] += take * 0.3

    # 4. Neutral venue: strip residual home advantage baked into club-form xG
    if home_ctx.is_neutral_venue:
        take = p["home"] * 0.05
        p["home"] -= take
        p["away"] += take * 0.7
        p["draw"] += take * 0.3

    total = sum(p.values())
    return {k: round(v / total, 4) for k, v in p.items()}


if __name__ == "__main__":
    base = {"home": 0.48, "draw": 0.26, "away": 0.26}
    h = TournamentContext(stage="knockout", rest_days=3, travel_km=2000)
    a = TournamentContext(stage="knockout", rest_days=5)
    print("adjusted:", tournament_adjust(base, h, a))
    with get_conn() as conn:
        games = query(conn, """select event_id from games
                               where status='scheduled' limit 5""")
        for g in games:
            for s in detect_anomalies(conn, g["event_id"]):
                if s.steam or s.rlm:
                    print(g["event_id"], s)

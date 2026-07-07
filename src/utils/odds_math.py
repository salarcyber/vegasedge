"""MODULE 3 — The Vegas-Beating Math.

Every formula the betting engine needs:
  * American <-> decimal odds conversion
  * Implied probability and vig removal (multiplicative + power devig)
  * Expected Value
  * Fractional Kelly staking
  * Poisson score-matrix models for soccer / hockey

All pure functions, fully unit-testable, no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------- conversions

def american_to_decimal(american: int | float) -> float:
    """+150 -> 2.50 ; -200 -> 1.50"""
    a = float(american)
    return 1 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def decimal_to_american(dec: float) -> int:
    return round((dec - 1) * 100) if dec >= 2.0 else round(-100 / (dec - 1))


def implied_prob(dec: float) -> float:
    """Raw implied probability — still contains the bookmaker's vig."""
    return 1.0 / dec


# ---------------------------------------------------------------- vig removal
# The book's probabilities sum to >1 (the overround). To recover the market's
# true consensus probability, normalize it out. Multiplicative devig is the
# standard; power devig better handles favourite-longshot bias on lopsided lines.

def devig_multiplicative(decimals: list[float]) -> list[float]:
    raw = [implied_prob(d) for d in decimals]
    total = sum(raw)
    return [p / total for p in raw]


def devig_power(decimals: list[float], tol: float = 1e-10) -> list[float]:
    """Find k such that sum(p_i^k) = 1, via bisection."""
    raw = [implied_prob(d) for d in decimals]
    lo, hi = 0.5, 3.0
    while hi - lo > tol:
        k = (lo + hi) / 2
        s = sum(p**k for p in raw)
        if s > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    powered = [p**k for p in raw]
    total = sum(powered)
    return [p / total for p in powered]


# ---------------------------------------------------------------- EV & Kelly

def expected_value(true_prob: float, dec_odds: float) -> float:
    """EV per $1 staked. Positive => value bet.

    EV = p * (dec - 1) - (1 - p)  ==  p * dec - 1
    """
    return true_prob * dec_odds - 1.0


def kelly_fraction(true_prob: float, dec_odds: float, multiplier: float = 0.25) -> float:
    """Fractional Kelly stake as a fraction of bankroll.

    Full Kelly: f* = (p*b - q) / b, with b = dec - 1, q = 1 - p.
    Full Kelly assumes your probability estimate is exact — it never is, and
    overbetting Kelly is ruinous, so we default to quarter-Kelly. Capped at 5%
    of bankroll as a hard safety rail.
    """
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - true_prob
    f = (true_prob * b - q) / b
    return max(0.0, min(f * multiplier, 0.05))


@dataclass
class BetEvaluation:
    true_prob: float
    market_prob: float          # devigged consensus
    best_decimal: float
    ev_pct: float
    edge_pct: float             # true_prob - market_prob, in points
    kelly_frac: float
    stake: float
    is_value: bool


def evaluate_bet(
    true_prob: float,
    best_decimal: float,
    market_decimals: list[float],
    bankroll: float,
    min_ev: float = 0.02,
    kelly_multiplier: float = 0.25,
    max_ev: float = 0.20,
) -> BetEvaluation:
    """The exact value-bet gate: flag only when EV clears `min_ev` (2% default —
    demanding a margin above zero absorbs model error and beats breakage) and
    stays under `max_ev` (a 20%+ 'edge' over an efficient market means the
    model is missing something the market knows — treat as error, not value)."""
    market_prob = devig_power(market_decimals)[0] if market_decimals else implied_prob(best_decimal)
    ev = expected_value(true_prob, best_decimal)
    is_value = min_ev < ev <= max_ev
    kf = kelly_fraction(true_prob, best_decimal, kelly_multiplier) if is_value else 0.0
    return BetEvaluation(
        true_prob=round(true_prob, 4),
        market_prob=round(market_prob, 4),
        best_decimal=best_decimal,
        ev_pct=round(ev * 100, 2),
        edge_pct=round((true_prob - market_prob) * 100, 2),
        kelly_frac=round(kf, 4),
        stake=round(bankroll * kf, 2),
        is_value=is_value,
    )


# ---------------------------------------------------------------- Poisson models
# For low-scoring sports (soccer, hockey) model each side's score as Poisson(λ)
# and derive every market (1X2, totals, exact score, BTTS) from the joint matrix.

def poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def score_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> list[list[float]]:
    """P(home=i, away=j) assuming independence. lam_* come from xG-based models."""
    home = [poisson_pmf(lam_home, i) for i in range(max_goals + 1)]
    away = [poisson_pmf(lam_away, j) for j in range(max_goals + 1)]
    return [[home[i] * away[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]


def match_probs(lam_home: float, lam_away: float) -> dict[str, float]:
    """1X2 + common totals from the Poisson matrix."""
    m = score_matrix(lam_home, lam_away)
    n = len(m)
    home = sum(m[i][j] for i in range(n) for j in range(n) if i > j)
    draw = sum(m[i][i] for i in range(n))
    away = sum(m[i][j] for i in range(n) for j in range(n) if i < j)
    over_25 = sum(m[i][j] for i in range(n) for j in range(n) if i + j >= 3)
    btts = sum(m[i][j] for i in range(1, n) for j in range(1, n))
    return {
        "home": home, "draw": draw, "away": away,
        "over_2.5": over_25, "under_2.5": 1 - over_25, "btts": btts,
    }


def expected_lambdas(
    xg_for_home: float, xg_against_home: float,
    xg_for_away: float, xg_against_away: float,
    league_avg_goals: float, home_adv: float = 1.15,
) -> tuple[float, float]:
    """Turn team xG rates into match-specific Poisson means.

    attack_strength = team xG for / league avg;  defence = team xG against / league avg.
    λ_home = league_avg * home_attack * away_defence * home_advantage
    """
    avg = league_avg_goals / 2  # per team
    lam_h = avg * (xg_for_home / avg) * (xg_against_away / avg) * home_adv
    lam_a = avg * (xg_for_away / avg) * (xg_against_home / avg)
    return lam_h, lam_a


if __name__ == "__main__":
    # sanity demo: model says 55% on a +105 line
    dec = american_to_decimal(105)
    ev = expected_value(0.55, dec)
    print(f"+105 @ p=0.55  EV={ev:+.2%}  kelly(quarter)={kelly_fraction(0.55, dec):.2%}")
    print("Poisson 1X2 (lambda 1.6 vs 1.1):", {k: round(v, 3) for k, v in match_probs(1.6, 1.1).items()})

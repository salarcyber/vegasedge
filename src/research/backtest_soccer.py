"""Honest backtest: our Poisson value engine vs real closing odds.

Data: football-data.co.uk free CSVs (results + Pinnacle/Bet365/market-best
closing prices). We replay each season chronologically:

  * team form = rolling last-15 goals for/against (point-in-time, like prod)
  * probabilities via the SAME expected_lambdas + match_probs as production
  * bet when 2% < EV <= 20% at the best available closing price (MaxC*)
  * de-vig market consensus from the average closing price (AvgC*)
  * quarter-Kelly staking capped at 5%, plus flat 1-unit ROI for a
    path-independent read

This bets CLOSING lines — the hardest test there is. Beating the close at
+ROI over 4,000+ matches would be extraordinary; the realistic pass mark is
"small negative, near the vig" (i.e., the model is close to sharp), with the
real-world edge coming from beating openers/soft books, not the close.

Run: python -m src.research.backtest_soccer
"""
from __future__ import annotations

import io

import httpx
import pandas as pd

from src.utils.odds_math import expected_lambdas, match_probs

LEAGUES = {"E0": "EPL", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga"}
SEASONS = ["2324", "2425", "2526"]
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

MIN_EV, MAX_EV = 0.02, 0.20
KELLY_MULT, KELLY_CAP = 0.25, 0.05


def load_csv(client: httpx.Client, season: str, code: str) -> pd.DataFrame | None:
    try:
        r = client.get(BASE.format(season=season, code=code), timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        return df
    except Exception as e:
        print(f"  {code} {season}: download failed ({e})")
        return None


def closing(df: pd.DataFrame, row, side: str) -> tuple[float | None, float | None]:
    """(best price, consensus price) for side in H/D/A from closing columns."""
    best = None
    for col in (f"MaxC{side}", f"PSC{side}", f"B365C{side}", f"B365{side}"):
        if col in df.columns and pd.notna(row.get(col)):
            best = float(row[col])
            break
    avg = None
    for col in (f"AvgC{side}", f"PSC{side}", f"B365C{side}", f"B365{side}"):
        if col in df.columns and pd.notna(row.get(col)):
            avg = float(row[col])
            break
    return best, avg


def backtest_league(df: pd.DataFrame, league: str, season: str) -> list[dict]:
    from collections import defaultdict, deque

    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=15))
    total_goals, n_games = 0.0, 0
    bets = []
    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        hh, ah = hist[h], hist[a]
        league_avg = (total_goals / n_games) if n_games >= 30 else 2.7
        if len(hh) >= 8 and len(ah) >= 8:
            hgf = sum(x[0] for x in hh) / len(hh)
            hga = sum(x[1] for x in hh) / len(hh)
            agf = sum(x[0] for x in ah) / len(ah)
            aga = sum(x[1] for x in ah) / len(ah)
            lam_h, lam_a = expected_lambdas(hgf, hga, agf, aga, league_avg)
            probs = match_probs(lam_h, lam_a)
            actual = "H" if row["FTHG"] > row["FTAG"] else \
                     "A" if row["FTAG"] > row["FTHG"] else "D"
            for side, p in (("H", probs["home"]), ("D", probs["draw"]), ("A", probs["away"])):
                best, _ = closing(df, row, side)
                if not best or best <= 1.01:
                    continue
                ev = p * best - 1
                if MIN_EV < ev <= MAX_EV:
                    kf = min(max((p * (best - 1) - (1 - p)) / (best - 1), 0) * KELLY_MULT,
                             KELLY_CAP)
                    bets.append({"league": league, "season": season, "side": side,
                                 "odds": best, "prob": p, "ev": ev, "kelly": kf,
                                 "won": side == actual})
        # update history AFTER evaluating (no lookahead)
        hist[h].append((row["FTHG"], row["FTAG"]))
        hist[a].append((row["FTAG"], row["FTHG"]))
        total_goals += row["FTHG"] + row["FTAG"]
        n_games += 1
    return bets


def summarize(bets: list[dict], label: str) -> None:
    if not bets:
        print(f"{label:>22}: no qualifying bets")
        return
    n = len(bets)
    wins = sum(b["won"] for b in bets)
    flat = sum((b["odds"] - 1) if b["won"] else -1 for b in bets)
    bank = 1000.0
    for b in bets:
        stake = bank * b["kelly"]
        bank += stake * (b["odds"] - 1) if b["won"] else -stake
    print(f"{label:>22}: {n:4d} bets | hit {wins/n:5.1%} | avg odds "
          f"{sum(b['odds'] for b in bets)/n:4.2f} | flat ROI {flat/n:+6.2%} | "
          f"kelly bank $1000 -> ${bank:,.0f}")


def main() -> None:
    client = httpx.Client(timeout=30, follow_redirects=True)
    all_bets: list[dict] = []
    for code, league in LEAGUES.items():
        for season in SEASONS:
            df = load_csv(client, season, code)
            if df is None:
                continue
            bets = backtest_league(df, league, season)
            all_bets.extend(bets)
            summarize(bets, f"{league} {season}")
    print("-" * 88)
    summarize(all_bets, "ALL (vs CLOSING line)")
    by_ev_low = [b for b in all_bets if b["ev"] <= 0.08]
    by_ev_high = [b for b in all_bets if b["ev"] > 0.08]
    summarize(by_ev_low, "edges 2-8% only")
    summarize(by_ev_high, "edges 8-20% only")


if __name__ == "__main__":
    main()

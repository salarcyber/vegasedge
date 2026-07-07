"""One-time MLB history backfill for model training.

  * Final scores for past seasons from ESPN's free scoreboard JSON (day by day).
  * Season-level team metrics (wRC+, wOBA, FIP, ERA) from FanGraphs via
    pybaseball, stored with as_of = season start so training rows join to the
    right season's numbers.

Honest caveat: season-aggregate stats include information from *after* each
game (mild lookahead). Good enough to bootstrap a v1 model; the weekly cloud
retrain gradually replaces this with rolling, point-in-time data collected by
the live pipeline.

Run: python -m src.ingestion.backfill_mlb 2024 2025 2026
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import httpx

from src.utils.db import get_conn, upsert

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

SEASON_WINDOWS = {  # opening day .. last regular-season day (approx, safe bounds)
    2024: (date(2024, 3, 28), date(2024, 9, 30)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
    2026: (date(2026, 3, 25), min(date(2026, 10, 1), date.today() - timedelta(days=1))),
}

# FanGraphs abbreviations -> full names used by ESPN / The Odds API
FG_TO_FULL = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Athletics", "ATH": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres",
    "SEA": "Seattle Mariners", "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays", "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}


def backfill_games(conn, season: int) -> int:
    start, end = SEASON_WINDOWS[season]
    d, stored = start, 0
    client = httpx.Client(timeout=25)
    while d <= end:
        try:
            r = client.get(SCOREBOARD, params={"dates": d.strftime("%Y%m%d")})
            r.raise_for_status()
            games = []
            for ev in r.json().get("events", []):
                comp = ev["competitions"][0]
                if not comp["status"]["type"]["completed"]:
                    continue
                home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
                away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
                hname, aname = home["team"]["displayName"], away["team"]["displayName"]
                upsert(conn, "teams",
                       [{"team_id": f"mlb_{hname}", "sport": "mlb", "name": hname},
                        {"team_id": f"mlb_{aname}", "sport": "mlb", "name": aname}],
                       ["team_id"])
                games.append({
                    "event_id": f"espn_mlb_{ev['id']}", "sport": "mlb", "league": "mlb",
                    "commence_time": ev["date"], "home_team_id": f"mlb_{hname}",
                    "away_team_id": f"mlb_{aname}", "status": "final",
                    "home_score": int(home["score"]), "away_score": int(away["score"]),
                })
            upsert(conn, "games", games, ["event_id"])
            stored += len(games)
        except Exception as e:
            print(f"  {d}: failed ({e}) — continuing")
        if d.day == 1:
            print(f"  {season}: through {d}, {stored} games so far")
            conn.commit()
        d += timedelta(days=1)
        time.sleep(0.15)  # be polite to ESPN
    print(f"[backfill] {season}: {stored} final games stored")
    return stored


def fetch_mlb_team_stats(season: int) -> dict[str, dict]:
    """Season team stats from MLB's official free StatsAPI (no key needed).
    Returns {full team name: {ops, runs_pg, era, whip}}."""
    out: dict[str, dict] = {}
    for group in ("hitting", "pitching"):
        r = httpx.get(
            "https://statsapi.mlb.com/api/v1/teams/stats",
            params={"season": season, "group": group, "stats": "season", "sportIds": 1},
            timeout=30,
        )
        r.raise_for_status()
        for split in r.json()["stats"][0]["splits"]:
            name = split["team"]["name"]
            s = split["stat"]
            m = out.setdefault(name, {})
            if group == "hitting":
                m["ops"] = float(s["ops"])
                m["runs_pg"] = round(int(s["runs"]) / max(int(s["gamesPlayed"]), 1), 3)
            else:
                m["era"] = float(s["era"])
                m["whip"] = float(s["whip"])
    return out


def backfill_metrics(conn, season: int) -> None:
    stats = fetch_mlb_team_stats(season)
    as_of = SEASON_WINDOWS[season][0]
    rows = [{"team_id": f"mlb_{name}", "as_of": as_of, "metrics": m}
            for name, m in stats.items()]
    upsert(conn, "teams", [{"team_id": r["team_id"], "sport": "mlb",
                            "name": r["team_id"][4:]} for r in rows], ["team_id"])
    upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
    print(f"[backfill] {season}: metrics for {len(rows)} teams (as_of {as_of})")


if __name__ == "__main__":
    seasons = [int(s) for s in sys.argv[1:]] or [2024, 2025, 2026]
    with get_conn() as conn:
        for s in seasons:
            backfill_metrics(conn, s)
            backfill_games(conn, s)

"""MODULE 1 — Live odds ingestion from The Odds API (free 500 credits/mo).

Run hourly by GitHub Actions. Snapshots are append-only so we retain the full
line-movement history (that history *is* the sharp-money signal in Module 4).

Credit budgeting: we only hit the API for sports that have games in the next
48h, and we tag the first snapshot per (event, market, outcome) as the opener.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import httpx

from src.utils.db import get_conn, insert_many, query, upsert
from src.utils.odds_math import american_to_decimal

BASE = "https://api.the-odds-api.com/v4"

# sport keys per The Odds API docs
SPORTS = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "epl": "soccer_epl",
    "worldcup": "soccer_fifa_world_cup",
    "ucl": "soccer_uefa_champs_league",
    "laliga": "soccer_spain_la_liga",
    "seriea": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "mls": "soccer_usa_mls",
}
MARKETS = "h2h,totals"   # 2 credits/sport/pull; add spreads when the engine prices them
REGIONS = "us"

# CLI name -> sport value stored in the DB (shared with the backfiller)
DB_SPORT = {"epl": "soccer_epl", "laliga": "soccer_laliga", "seriea": "soccer_seriea",
            "bundesliga": "soccer_bundesliga", "ucl": "soccer_ucl", "mls": "soccer_mls"}


def fetch_odds(sport_key: str, api_key: str) -> list[dict]:
    r = httpx.get(
        f"{BASE}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": REGIONS,
            "markets": MARKETS,
            "oddsFormat": "american",
        },
        timeout=30,
    )
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"[odds] {sport_key}: {len(r.json())} events, {remaining} credits left")
    return r.json()


def ingest_sport(conn, sport: str, sport_key: str, api_key: str) -> None:
    sport = DB_SPORT.get(sport, sport)
    events = fetch_odds(sport_key, api_key)
    horizon = datetime.now(timezone.utc) + timedelta(hours=48)

    games, snapshots = [], []
    for ev in events:
        commence = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if commence > horizon:
            continue
        home_id = f"{sport}_{ev['home_team']}"
        away_id = f"{sport}_{ev['away_team']}"
        upsert(conn, "teams",
               [{"team_id": home_id, "sport": sport, "name": ev["home_team"]},
                {"team_id": away_id, "sport": sport, "name": ev["away_team"]}],
               ["team_id"])
        games.append({
            "event_id": ev["id"], "sport": sport, "league": sport_key,
            "commence_time": commence, "home_team_id": home_id,
            "away_team_id": away_id, "raw": ev,
        })
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                for out in mkt.get("outcomes", []):
                    snapshots.append({
                        "event_id": ev["id"],
                        "bookmaker": bk["key"],
                        "market": mkt["key"],
                        "outcome": out["name"],
                        "line": out.get("point"),
                        "price_american": int(out["price"]),
                        "price_decimal": round(american_to_decimal(out["price"]), 4),
                    })

    upsert(conn, "games", games, ["event_id"])
    insert_many(conn, "odds_snapshots", snapshots)

    # mark openers: first snapshot ever seen for each (event, market, outcome, book)
    with conn.cursor() as cur:
        cur.execute("""
            update odds_snapshots s set is_opening = true
            where s.id in (
                select distinct on (event_id, bookmaker, market, outcome) id
                from odds_snapshots order by event_id, bookmaker, market, outcome, captured_at asc
            ) and not s.is_opening
        """)
    print(f"[odds] {sport}: stored {len(games)} games, {len(snapshots)} snapshots")


def main(sports: list[str]) -> None:
    api_key = os.environ["ODDS_API_KEY"]
    with get_conn() as conn:
        for s in sports:
            if s not in SPORTS:
                print(f"[odds] unknown sport {s}, skipping")
                continue
            try:
                ingest_sport(conn, s, SPORTS[s], api_key)
            except httpx.HTTPStatusError as e:
                print(f"[odds] {s} failed: {e.response.status_code} {e.response.text[:200]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nba", "nfl"])

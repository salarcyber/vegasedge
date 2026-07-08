"""MLB starting pitchers — the single biggest MLB feature.

Source: MLB's official free StatsAPI (no key).
  * schedule?hydrate=probablePitcher  -> who starts each game
  * people/{id}/stats?stats=season    -> that pitcher's ERA / WHIP / K/9

Starters are stored on game_context.notes as {"home_sp": {...}, "away_sp": {...}}
and become d_sp_* features in the model (features/preprocess.py).

CLI:
  python -m src.ingestion.pitchers current          # today + tomorrow (hourly job)
  python -m src.ingestion.pitchers backfill 2024 2025 2026
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import httpx

from src.utils.db import get_conn, query, upsert

API = "https://statsapi.mlb.com/api/v1"
_stats_cache: dict[tuple[int, int], dict | None] = {}


def pitcher_stats(client: httpx.Client, pid: int, season: int) -> dict | None:
    if (pid, season) in _stats_cache:
        return _stats_cache[(pid, season)]
    out = None
    try:
        r = client.get(f"{API}/people/{pid}/stats",
                       params={"stats": "season", "group": "pitching", "season": season})
        splits = r.json()["stats"][0]["splits"]
        if splits:
            s = splits[0]["stat"]
            out = {"era": float(s["era"]), "whip": float(s["whip"]),
                   "k9": float(s["strikeoutsPer9Inn"]),
                   "ip": float(s["inningsPitched"])}
    except Exception:
        out = None
    _stats_cache[(pid, season)] = out
    return out


def day_starters(client: httpx.Client, d: date) -> list[dict]:
    r = client.get(f"{API}/schedule", params={
        "sportId": 1, "date": d.strftime("%Y-%m-%d"), "hydrate": "probablePitcher"})
    r.raise_for_status()
    out = []
    for day in r.json().get("dates", []):
        for g in day.get("games", []):
            teams = g["teams"]
            row = {"date": d,
                   "home": teams["home"]["team"]["name"],
                   "away": teams["away"]["team"]["name"]}
            for side in ("home", "away"):
                pp = teams[side].get("probablePitcher")
                if pp:
                    row[f"{side}_sp"] = {"id": pp["id"], "name": pp["fullName"]}
            out.append(row)
    return out


def attach_starters(conn, rows: list[dict], client: httpx.Client) -> int:
    n = 0
    for row in rows:
        if "home_sp" not in row and "away_sp" not in row:
            continue
        game = query(conn, """
            select event_id, coalesce((select notes from game_context gc
                                       where gc.event_id = g.event_id), '{}'::jsonb) notes
            from games g
            where g.sport = 'mlb' and g.home_team_id = %s and g.away_team_id = %s
              and g.commence_time::date between %s and %s
            order by g.commence_time limit 1
        """, (f"mlb_{row['home']}", f"mlb_{row['away']}",
              row["date"], row["date"] + timedelta(days=1)))
        if not game:
            continue
        notes = game[0]["notes"] or {}
        for side in ("home", "away"):
            sp = row.get(f"{side}_sp")
            if not sp:
                continue
            stats = pitcher_stats(client, sp["id"], row["date"].year) or {}
            notes[f"{side}_sp"] = {"name": sp["name"], **stats}
            upsert(conn, "players", [{
                "player_id": f"mlb_{sp['id']}", "sport": "mlb",
                "name": sp["name"], "position": "SP",
            }], ["player_id"])
        upsert(conn, "game_context",
               [{"event_id": game[0]["event_id"], "notes": notes}], ["event_id"])
        n += 1
    return n


def run_current(conn) -> None:
    client = httpx.Client(timeout=25)
    total = 0
    for d in (date.today(), date.today() + timedelta(days=1)):
        total += attach_starters(conn, day_starters(client, d), client)
    print(f"[pitchers] starters attached for {total} upcoming games")


def run_backfill(conn, seasons: list[int]) -> None:
    from src.ingestion.backfill_mlb import SEASON_WINDOWS

    client = httpx.Client(timeout=25)
    for season in seasons:
        start, end = SEASON_WINDOWS[season]
        d, n = start, 0
        while d <= end:
            try:
                n += attach_starters(conn, day_starters(client, d), client)
            except Exception as e:
                print(f"[pitchers] {d} failed: {e}")
            if d.day == 1:
                conn.commit()
                print(f"[pitchers] {season}: through {d}, {n} games")
            d += timedelta(days=1)
            time.sleep(0.1)
        print(f"[pitchers] {season}: starters on {n} games "
              f"({len(_stats_cache)} pitcher-season stats cached)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"
    with get_conn() as conn:
        if mode == "backfill":
            run_backfill(conn, [int(s) for s in sys.argv[2:]] or [2024, 2025, 2026])
        else:
            run_current(conn)

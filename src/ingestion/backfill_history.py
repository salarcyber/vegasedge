"""Multi-sport history backfill (NBA, NFL, NHL, EPL) for model training.

Same pattern as backfill_mlb: day-by-day final scores from ESPN's free
scoreboard JSON + one season-level metrics snapshot per season (stored at
season start; mild lookahead, acceptable to bootstrap v1 — the live pipeline
replaces it with point-in-time data going forward).

Run: python -m src.ingestion.backfill_history nba nfl nhl epl
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import httpx

from src.utils.db import get_conn, upsert

ESPN = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"

SPORTS = {
    "nba": {"path": "basketball/nba",
            "seasons": [(date(2023, 10, 24), date(2024, 6, 20)),
                        (date(2024, 10, 22), date(2025, 6, 20)),
                        (date(2025, 10, 21), date(2026, 6, 22))]},
    "nfl": {"path": "football/nfl",
            "seasons": [(date(2023, 9, 7), date(2024, 2, 12)),
                        (date(2024, 9, 5), date(2025, 2, 10)),
                        (date(2025, 9, 4), date(2026, 2, 9))]},
    "nhl": {"path": "hockey/nhl",
            "seasons": [(date(2023, 10, 10), date(2024, 6, 25)),
                        (date(2024, 10, 8), date(2025, 6, 25)),
                        (date(2025, 10, 7), date(2026, 6, 25))]},
    "epl": {"path": "soccer/eng.1",
            "seasons": [(date(2023, 8, 11), date(2024, 5, 20)),
                        (date(2024, 8, 16), date(2025, 5, 26)),
                        (date(2025, 8, 15), date(2026, 5, 25))]},
}
# note: 'epl' games are stored with sport='soccer_epl' to match the odds ingester
DB_SPORT = {"epl": "soccer_epl"}


def backfill_games(conn, sport: str) -> int:
    cfg = SPORTS[sport]
    db_sport = DB_SPORT.get(sport, sport)
    client = httpx.Client(timeout=25)
    total = 0
    for start, end in cfg["seasons"]:
        d, stored = start, 0
        while d <= min(end, date.today() - timedelta(days=1)):
            try:
                r = client.get(ESPN.format(path=cfg["path"]),
                               params={"dates": d.strftime("%Y%m%d")})
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
                           [{"team_id": f"{db_sport}_{hname}", "sport": db_sport, "name": hname},
                            {"team_id": f"{db_sport}_{aname}", "sport": db_sport, "name": aname}],
                           ["team_id"])
                    games.append({
                        "event_id": f"espn_{sport}_{ev['id']}", "sport": db_sport,
                        "league": sport, "commence_time": ev["date"],
                        "home_team_id": f"{db_sport}_{hname}",
                        "away_team_id": f"{db_sport}_{aname}", "status": "final",
                        "home_score": int(home["score"]), "away_score": int(away["score"]),
                    })
                upsert(conn, "games", games, ["event_id"])
                stored += len(games)
            except Exception as e:
                print(f"  {sport} {d}: failed ({e}) — continuing")
            if d.day == 1:
                conn.commit()
                print(f"  {sport}: through {d}, +{stored} this season")
            d += timedelta(days=1)
            time.sleep(0.15)
        total += stored
        print(f"[backfill] {sport} season {start.year}-{end.year}: {stored} games")
    return total


# ------------------------------------------------------- per-season metrics

def metrics_nba(conn) -> None:
    from nba_api.stats.endpoints import leaguedashteamstats

    for season, as_of in (("2023-24", date(2023, 10, 24)),
                          ("2024-25", date(2024, 10, 22)),
                          ("2025-26", date(2025, 10, 21))):
        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame", timeout=60).get_data_frames()[0]
        rows = [{"team_id": f"nba_{r['TEAM_NAME']}", "as_of": as_of,
                 "metrics": {"net_rating": float(r["NET_RATING"]),
                             "off_rating": float(r["OFF_RATING"]),
                             "def_rating": float(r["DEF_RATING"]),
                             "pace": float(r["PACE"]), "efg_pct": float(r["EFG_PCT"]),
                             "tov_pct": float(r["TM_TOV_PCT"])}}
                for _, r in adv.iterrows()]
        upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
        print(f"[metrics] nba {season}: {len(rows)} teams")
        time.sleep(2)


NFL_ABBR = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def metrics_nfl(conn) -> None:
    import nfl_data_py as nfl

    for season, as_of in ((2023, date(2023, 9, 7)), (2024, date(2024, 9, 5)),
                          (2025, date(2025, 9, 4))):
        pbp = nfl.import_pbp_data([season], columns=["posteam", "defteam", "epa", "success"])
        pbp = pbp.dropna(subset=["posteam", "epa"])
        off = pbp.groupby("posteam").agg(epa_off=("epa", "mean"), success_off=("success", "mean"))
        deff = pbp.groupby("defteam").agg(epa_def=("epa", "mean"), success_def=("success", "mean"))
        merged = off.join(deff)
        rows = [{"team_id": f"nfl_{NFL_ABBR[t]}", "as_of": as_of,
                 "metrics": {k: round(float(v), 4) for k, v in m.items()}}
                for t, m in merged.iterrows() if t in NFL_ABBR]
        upsert(conn, "teams", [{"team_id": r["team_id"], "sport": "nfl",
                                "name": r["team_id"][4:]} for r in rows], ["team_id"])
        upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
        print(f"[metrics] nfl {season}: {len(rows)} teams (abbrev ids)")


def metrics_nhl(conn) -> None:
    for end_date, as_of in ((date(2024, 4, 18), date(2023, 10, 10)),
                            (date(2025, 4, 17), date(2024, 10, 8)),
                            (date(2026, 4, 16), date(2025, 10, 7))):
        r = httpx.get(f"https://api-web.nhle.com/v1/standings/{end_date}", timeout=30)
        r.raise_for_status()
        rows = []
        for t in r.json()["standings"]:
            gp = t["gamesPlayed"] or 1
            rows.append({"team_id": f"nhl_{t['teamName']['default']}", "as_of": as_of,
                         "metrics": {"gf_per_game": round(t["goalFor"] / gp, 3),
                                     "ga_per_game": round(t["goalAgainst"] / gp, 3),
                                     "point_pct": t["pointPctg"]}})
        upsert(conn, "teams", [{"team_id": x["team_id"], "sport": "nhl",
                                "name": x["team_id"][4:]} for x in rows], ["team_id"])
        upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
        print(f"[metrics] nhl season ending {end_date}: {len(rows)} teams")


def metrics_epl(conn) -> None:
    from src.ingestion.stats_ingest import ingest_soccer_understat
    # current season only via the live ingester; historical understat seasons:
    import json
    import re
    for season, as_of in (("2023", date(2023, 8, 11)), ("2024", date(2024, 8, 16)),
                          ("2025", date(2025, 8, 15))):
        try:
            r = httpx.get(f"https://understat.com/league/EPL/{season}",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            m = re.search(r"teamsData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
            data = json.loads(m.group(1).encode().decode("unicode_escape"))
            rows = []
            for team in data.values():
                hist = team["history"]
                n = len(hist) or 1
                rows.append({"team_id": f"soccer_epl_{team['title']}", "as_of": as_of,
                             "metrics": {"xg_for": round(sum(h["xG"] for h in hist) / n, 3),
                                         "xg_against": round(sum(h["xGA"] for h in hist) / n, 3)}})
            upsert(conn, "teams", [{"team_id": x["team_id"], "sport": "soccer_epl",
                                    "name": x["team_id"][11:]} for x in rows], ["team_id"])
            upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
            print(f"[metrics] epl {season}: {len(rows)} teams")
        except Exception as e:
            print(f"[metrics] epl {season} failed: {e}")


METRICS = {"nba": metrics_nba, "nfl": metrics_nfl, "nhl": metrics_nhl, "epl": metrics_epl}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(SPORTS)
    with get_conn() as conn:
        for t in targets:
            try:
                METRICS[t](conn)
            except Exception as e:
                print(f"[metrics] {t} FAILED: {e}")
            try:
                backfill_games(conn, t)
            except Exception as e:
                print(f"[backfill] {t} FAILED: {e}")
    print("[backfill] history run complete")

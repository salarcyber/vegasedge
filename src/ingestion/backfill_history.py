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
    "laliga": {"path": "soccer/esp.1",
               "seasons": [(date(2023, 8, 11), date(2024, 5, 27)),
                           (date(2024, 8, 15), date(2025, 5, 26)),
                           (date(2025, 8, 15), date(2026, 5, 25))]},
    "seriea": {"path": "soccer/ita.1",
               "seasons": [(date(2023, 8, 19), date(2024, 5, 27)),
                           (date(2024, 8, 17), date(2025, 5, 26)),
                           (date(2025, 8, 23), date(2026, 5, 25))]},
    "bundesliga": {"path": "soccer/ger.1",
                   "seasons": [(date(2023, 8, 18), date(2024, 5, 19)),
                               (date(2024, 8, 23), date(2025, 5, 18)),
                               (date(2025, 8, 22), date(2026, 5, 17))]},
    "ucl": {"path": "soccer/uefa.champions",
            "seasons": [(date(2023, 9, 19), date(2024, 6, 2)),
                        (date(2024, 9, 17), date(2025, 6, 1)),
                        (date(2025, 9, 16), date(2026, 6, 1))]},
    "mls": {"path": "soccer/usa.1",
            "seasons": [(date(2024, 2, 21), date(2024, 12, 8)),
                        (date(2025, 2, 22), date(2025, 12, 7)),
                        (date(2026, 2, 21), date(2026, 12, 6))]},
    "worldcup": {"path": "soccer/fifa.world",
                 "seasons": [(date(2026, 6, 11), date(2026, 7, 19))]},
}
# CLI name -> sport value stored in the DB (must match the odds ingester)
DB_SPORT = {"epl": "soccer_epl", "laliga": "soccer_laliga", "seriea": "soccer_seriea",
            "bundesliga": "soccer_bundesliga", "ucl": "soccer_ucl", "mls": "soccer_mls",
            "worldcup": "worldcup"}


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


def metrics_from_games(conn, sport: str, db_sport: str, keys: tuple[str, str],
                       min_matches: int = 10) -> None:
    """Per-season scoring metrics computed from our own backfilled games —
    no external API to block or break. keys = (for_key, against_key)."""
    from src.utils.db import query

    for_key, against_key = keys
    for start, end in SPORTS[sport]["seasons"]:
        rows = query(conn, """
            with sides as (
              select home_team_id team_id, home_score pf, away_score pa
              from games where sport=%s and status='final'
                and commence_time::date between %s and %s
              union all
              select away_team_id, away_score, home_score
              from games where sport=%s and status='final'
                and commence_time::date between %s and %s
            )
            select team_id, count(*) n, avg(pf) pf_pg, avg(pa) pa_pg,
                   avg((pf > pa)::int) win_pct
            from sides group by team_id having count(*) >= %s
        """, (db_sport, start, end, db_sport, start, end, min_matches))
        metrics = [{"team_id": r["team_id"], "as_of": start,
                    "metrics": {for_key: round(float(r["pf_pg"]), 3),
                                against_key: round(float(r["pa_pg"]), 3),
                                "net_pg": round(float(r["pf_pg"] - r["pa_pg"]), 3),
                                "win_pct": round(float(r["win_pct"]), 3),
                                "matches": r["n"]}} for r in rows]
        upsert(conn, "team_metrics", metrics, ["team_id", "as_of"])
        print(f"[metrics] {sport} {start.year}-{end.year}: {len(metrics)} teams (from game data)")


def metrics_epl(conn) -> None:
    metrics_from_games(conn, "epl", "soccer_epl", ("gf_per_game", "ga_per_game"))


def metrics_nba_from_games(conn) -> None:
    metrics_from_games(conn, "nba", "nba", ("pf_pg", "pa_pg"))


def _soccer_metrics(sport: str, min_matches: int = 10):
    def run(conn):
        metrics_from_games(conn, sport, DB_SPORT.get(sport, sport),
                           ("gf_per_game", "ga_per_game"), min_matches)
    return run


METRICS = {
    "nba": metrics_nba_from_games, "nfl": metrics_nfl, "nhl": metrics_nhl,
    "epl": metrics_epl,
    "laliga": _soccer_metrics("laliga"), "seriea": _soccer_metrics("seriea"),
    "bundesliga": _soccer_metrics("bundesliga"), "ucl": _soccer_metrics("ucl", 4),
    "mls": _soccer_metrics("mls"),
    "worldcup": _soccer_metrics("worldcup", 3),  # short tournament, few matches
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(SPORTS)
    # one connection per sport so a dropped connection can't kill the whole run;
    # games before metrics so team rows exist for the FK
    for t in targets:
        try:
            with get_conn() as conn:
                backfill_games(conn, t)
        except Exception as e:
            print(f"[backfill] {t} FAILED: {e}")
        try:
            with get_conn() as conn:
                METRICS[t](conn)
        except Exception as e:
            print(f"[metrics] {t} FAILED: {e}")
    print("[backfill] history run complete")

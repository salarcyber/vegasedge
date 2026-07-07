"""MODULE 1/2 — Advanced team & player metrics from free sources.

  NBA   : nba_api (official stats.nba.com) — Net Rating, ORtg/DRtg, pace
  NFL   : nfl_data_py (nflverse pbp)       — EPA/play offense & defense, success rate
  MLB   : pybaseball (FanGraphs)           — wRC+, FIP, team wOBA
  NHL   : NHL public API                   — goals for/against rates, PP%, PK%
  Soccer: Understat scrape                 — xG for/against per match

Each ingester writes one row per team into team_metrics(as_of=today, metrics=jsonb).
Run daily. Every source below is completely free.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date

import httpx

from src.utils.db import get_conn, upsert

TODAY = date.today()


def ingest_nba(conn) -> None:
    from nba_api.stats.endpoints import leaguedashteamstats

    adv = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame"
    ).get_data_frames()[0]
    rows = []
    for _, r in adv.iterrows():
        rows.append({
            "team_id": f"nba_{r['TEAM_NAME']}",
            "as_of": TODAY,
            "metrics": {
                "net_rating": float(r["NET_RATING"]),
                "off_rating": float(r["OFF_RATING"]),
                "def_rating": float(r["DEF_RATING"]),
                "pace": float(r["PACE"]),
                "efg_pct": float(r["EFG_PCT"]),
                "tov_pct": float(r["TM_TOV_PCT"]),
                "reb_pct": float(r["REB_PCT"]),
            },
        })
    upsert(conn, "teams", [{"team_id": x["team_id"], "sport": "nba",
                            "name": x["team_id"][4:]} for x in rows], ["team_id"])
    upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
    print(f"[stats] nba: {len(rows)} teams")


def ingest_nfl(conn, season: int) -> None:
    import nfl_data_py as nfl
    import pandas as pd

    pbp = nfl.import_pbp_data([season], columns=[
        "posteam", "defteam", "epa", "success", "pass", "rush"
    ])
    pbp = pbp.dropna(subset=["posteam", "epa"])
    off = pbp.groupby("posteam").agg(epa_off=("epa", "mean"),
                                     success_off=("success", "mean"))
    deff = pbp.groupby("defteam").agg(epa_def=("epa", "mean"),
                                      success_def=("success", "mean"))
    merged = off.join(deff)
    rows = [{
        "team_id": f"nfl_{team}",
        "as_of": TODAY,
        "metrics": {k: round(float(v), 4) for k, v in m.items()},
    } for team, m in merged.iterrows()]
    upsert(conn, "teams", [{"team_id": x["team_id"], "sport": "nfl",
                            "name": x["team_id"][4:]} for x in rows], ["team_id"])
    upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
    print(f"[stats] nfl: {len(rows)} teams")


def ingest_mlb(conn, season: int) -> None:
    from pybaseball import team_batting, team_pitching

    bat = team_batting(season)
    pit = team_pitching(season)
    rows = []
    for _, r in bat.iterrows():
        team = r["Team"]
        p = pit[pit["Team"] == team]
        rows.append({
            "team_id": f"mlb_{team}",
            "as_of": TODAY,
            "metrics": {
                "wrc_plus": float(r["wRC+"]),
                "woba": float(r["wOBA"]),
                "fip": float(p["FIP"].iloc[0]) if len(p) else None,
                "era": float(p["ERA"].iloc[0]) if len(p) else None,
            },
        })
    upsert(conn, "teams", [{"team_id": x["team_id"], "sport": "mlb",
                            "name": x["team_id"][4:]} for x in rows], ["team_id"])
    upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
    print(f"[stats] mlb: {len(rows)} teams")


def ingest_nhl(conn) -> None:
    r = httpx.get("https://api-web.nhle.com/v1/standings/now", timeout=30)
    r.raise_for_status()
    rows = []
    for t in r.json()["standings"]:
        gp = t["gamesPlayed"] or 1
        rows.append({
            "team_id": f"nhl_{t['teamName']['default']}",
            "as_of": TODAY,
            "metrics": {
                "gf_per_game": round(t["goalFor"] / gp, 3),
                "ga_per_game": round(t["goalAgainst"] / gp, 3),
                "point_pct": t["pointPctg"],
                "l10_wins": t.get("l10Wins"),
            },
        })
    upsert(conn, "teams", [{"team_id": x["team_id"], "sport": "nhl",
                            "name": x["team_id"][4:]} for x in rows], ["team_id"])
    upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
    print(f"[stats] nhl: {len(rows)} teams")


def ingest_soccer_understat(conn, league: str = "EPL", season: str = "2025") -> None:
    """Understat embeds a JSON blob of per-team xG in the league page — scrape it."""
    r = httpx.get(
        f"https://understat.com/league/{league}/{season}",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    r.raise_for_status()
    m = re.search(r"teamsData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
    if not m:
        print("[stats] understat: page format changed, update the regex")
        return
    data = json.loads(m.group(1).encode().decode("unicode_escape"))
    rows = []
    for team in data.values():
        hist = team["history"]
        n = len(hist) or 1
        rows.append({
            "team_id": f"epl_{team['title']}",
            "as_of": TODAY,
            "metrics": {
                "xg_for": round(sum(h["xG"] for h in hist) / n, 3),
                "xg_against": round(sum(h["xGA"] for h in hist) / n, 3),
                "ppda": round(sum(h["ppda"]["att"] / max(h["ppda"]["def"], 1) for h in hist) / n, 2),
                "matches": n,
            },
        })
    upsert(conn, "teams", [{"team_id": x["team_id"], "sport": "soccer_epl",
                            "name": x["team_id"][4:]} for x in rows], ["team_id"])
    upsert(conn, "team_metrics", rows, ["team_id", "as_of"])
    print(f"[stats] {league}: {len(rows)} teams")


INGESTERS = {
    "nba": lambda c: ingest_nba(c),
    "nfl": lambda c: ingest_nfl(c, TODAY.year if TODAY.month >= 9 else TODAY.year - 1),
    "mlb": lambda c: ingest_mlb(c, TODAY.year),
    "nhl": lambda c: ingest_nhl(c),
    "epl": lambda c: ingest_soccer_understat(c),
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(INGESTERS)
    with get_conn() as conn:
        for t in targets:
            try:
                INGESTERS[t](conn)
            except Exception as e:  # one sport failing must not kill the pipeline
                print(f"[stats] {t} FAILED: {e}")

"""MODULE 7 — Hyper-Granular Deep Data Engine (all free sources).

Beyond box scores, this collects the marginal signals that soft lines miss:

  1. Weather anomalies  — Open-Meteo (free, no key): wind speed/direction,
     humidity, barometric pressure, precipitation at kickoff, at the venue.
  2. Travel fatigue     — haversine distance between the away team's previous
     venue and this venue, time zones crossed (circadian penalty), rest days.
  3. Sentiment          — Reddit public JSON of team subreddits + beat-writer
     RSS, scored with VADER. Gauges locker-room mood / chemistry noise.
  4. Situational splits — performance under a specific referee, on back-to-backs,
     after losses, on turf vs grass, computed from our own game history.
  5. Injuries           — ESPN's unauthenticated injuries JSON per league.

Each signal is stored in game_context / situational_splits and consumed as
model features in Module 2 (see features/preprocess.py FEATURE_WEIGHTS for
the priors on how heavily each is allowed to move the prediction).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import httpx

from src.utils.db import get_conn, query, upsert

UA = {"User-Agent": "Mozilla/5.0 (VegasEdge research; contact in repo)"}


# ------------------------------------------------------------------ 1. weather

def fetch_weather(lat: float, lon: float, kickoff_iso: str) -> dict | None:
    """Open-Meteo hourly forecast at the venue for the kickoff hour."""
    r = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,windspeed_10m,winddirection_10m,"
                      "relativehumidity_2m,surface_pressure,precipitation",
            "timezone": "UTC",
        },
        timeout=20,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    target = kickoff_iso[:13]  # match to the hour: 'YYYY-MM-DDTHH'
    for i, t in enumerate(h["time"]):
        if t.startswith(target):
            return {
                "temp_c": h["temperature_2m"][i],
                "wind_kph": h["windspeed_10m"][i],
                "wind_dir_deg": h["winddirection_10m"][i],
                "humidity": h["relativehumidity_2m"][i],
                "pressure_hpa": h["surface_pressure"][i],
                "precip_mm": h["precipitation"][i],
            }
    return None


# ------------------------------------------------------- 2. travel & fatigue

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def travel_context(conn, event_id: str) -> dict:
    """Rest days + travel km + time zones crossed for both teams, derived from
    each team's previous game in our own games table."""
    game = query(conn, """
        select g.*, ht.venue_lat hlat, ht.venue_lon hlon, ht.venue_tz htz
        from games g join teams ht on ht.team_id = g.home_team_id
        where g.event_id = %s
    """, (event_id,))
    if not game:
        return {}
    g = game[0]
    out: dict = {}
    for side in ("home", "away"):
        team = g[f"{side}_team_id"]
        prev = query(conn, """
            select g2.commence_time, t.venue_lat, t.venue_lon, t.venue_tz
            from games g2 join teams t on t.team_id = g2.home_team_id
            where (g2.home_team_id = %s or g2.away_team_id = %s)
              and g2.commence_time < %s
            order by g2.commence_time desc limit 1
        """, (team, team, g["commence_time"]))
        if prev and prev[0]["venue_lat"] and g["hlat"]:
            p = prev[0]
            out[f"{side}_travel_km"] = round(
                haversine_km(p["venue_lat"], p["venue_lon"], g["hlat"], g["hlon"]), 1)
            out[f"{side}_rest_days"] = (g["commence_time"] - p["commence_time"]).days
            # rough tz-crossed: 15 degrees longitude ≈ 1 zone
            out[f"tz_crossed_{side}"] = round(abs(p["venue_lon"] - g["hlon"]) / 15)
    return out


# ------------------------------------------------------------- 3. sentiment

TEAM_SUBREDDITS = {  # extend as needed
    "nba_Boston Celtics": "bostonceltics", "nba_Los Angeles Lakers": "lakers",
    "nfl_Kansas City Chiefs": "KansasCityChiefs", "nfl_Buffalo Bills": "buffalobills",
}


def reddit_sentiment(team_id: str, limit: int = 25) -> dict | None:
    """Score recent hot posts on the team subreddit with VADER. No API key —
    Reddit serves public JSON. This is a *noisy* signal; Module 2 caps its weight."""
    from nltk.sentiment import SentimentIntensityAnalyzer

    sub = TEAM_SUBREDDITS.get(team_id)
    if not sub:
        return None
    r = httpx.get(f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}",
                  headers=UA, timeout=20)
    if r.status_code != 200:
        return None
    posts = [c["data"]["title"] + " " + c["data"].get("selftext", "")[:300]
             for c in r.json()["data"]["children"]]
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(p)["compound"] for p in posts]
    return {
        "score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "n_docs": len(scores),
    }


# ------------------------------------------------------------- 4. situational splits

def compute_situational_splits(conn, sport: str) -> None:
    """Recompute rest-based splits from our own settled game history.
    Extend with referee splits once you log referees in game_context."""
    rows = query(conn, """
        with results as (
          select g.home_team_id team_id, g.home_rest_days rest,
                 (g.home_score > g.away_score)::int win
          from games g where g.sport = %s and g.status = 'final'
          union all
          select g.away_team_id, g.away_rest_days,
                 (g.away_score > g.home_score)::int
          from games g where g.sport = %s and g.status = 'final'
        )
        select team_id,
               case when rest <= 1 then 'rest:b2b' else 'rest:normal' end split_key,
               count(*) n, avg(win) win_pct
        from results where rest is not null
        group by 1, 2 having count(*) >= 5
    """, (sport, sport))
    upsert(conn, "situational_splits", [{
        "subject_id": r["team_id"], "split_key": r["split_key"],
        "n": r["n"], "value": {"win_pct": round(float(r["win_pct"]), 3)},
        "as_of": datetime.now(timezone.utc).date(),
    } for r in rows], ["subject_id", "split_key", "as_of"])
    print(f"[deep] {sport}: {len(rows)} situational splits")


# ------------------------------------------------------------- 5. injuries

ESPN_INJURY_URLS = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries",
}


def ingest_injuries(conn, sport: str) -> None:
    r = httpx.get(ESPN_INJURY_URLS[sport], headers=UA, timeout=20)
    r.raise_for_status()
    rows = []
    for team in r.json().get("injuries", []):
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {})
            pid = f"{sport}_{ath.get('id', ath.get('displayName', '?'))}"
            upsert(conn, "players", [{
                "player_id": pid, "sport": sport,
                "name": ath.get("displayName", "?"),
                "position": (ath.get("position") or {}).get("abbreviation"),
            }], ["player_id"])
            rows.append({
                "player_id": pid,
                "status": (inj.get("status") or "").lower(),
                "detail": inj.get("longComment", "")[:500],
                "source": "espn",
            })
    from src.utils.db import insert_many
    insert_many(conn, "injuries", rows)
    print(f"[deep] {sport}: {len(rows)} injury reports")


# ------------------------------------------------------------- orchestrator

def enrich_upcoming_games(conn) -> None:
    games = query(conn, """
        select g.event_id, g.sport, g.commence_time, g.home_team_id, g.away_team_id,
               t.venue_lat, t.venue_lon
        from games g join teams t on t.team_id = g.home_team_id
        where g.status = 'scheduled' and g.commence_time < now() + interval '48 hours'
    """)
    for g in games:
        ctx: dict = {"event_id": g["event_id"]}
        if g["venue_lat"]:
            try:
                ctx["weather"] = fetch_weather(
                    g["venue_lat"], g["venue_lon"], g["commence_time"].isoformat())
            except Exception as e:
                print(f"[deep] weather failed {g['event_id']}: {e}")
        trav = travel_context(conn, g["event_id"])
        ctx["tz_crossed_home"] = trav.get("tz_crossed_home", 0)
        ctx["tz_crossed_away"] = trav.get("tz_crossed_away", 0)
        sent = {}
        for side in ("home", "away"):
            s = None
            try:
                s = reddit_sentiment(g[f"{side}_team_id"])
            except Exception:
                pass
            if s:
                sent[side] = s
        if sent:
            ctx["sentiment"] = sent
        upsert(conn, "game_context", [ctx], ["event_id"])
        if trav:
            with conn.cursor() as cur:
                cur.execute("""update games set home_rest_days=%s, away_rest_days=%s,
                               home_travel_km=%s, away_travel_km=%s where event_id=%s""",
                            (trav.get("home_rest_days"), trav.get("away_rest_days"),
                             trav.get("home_travel_km"), trav.get("away_travel_km"),
                             g["event_id"]))
    print(f"[deep] enriched {len(games)} upcoming games")


if __name__ == "__main__":
    with get_conn() as conn:
        for sport in ("nba", "nfl", "mlb", "nhl"):
            try:
                ingest_injuries(conn, sport)
                compute_situational_splits(conn, sport)
            except Exception as e:
                print(f"[deep] {sport} failed: {e}")
        enrich_upcoming_games(conn)

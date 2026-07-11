"""MODULE 5 — LLM Reasoning Engine.

Feeds each flagged value bet (plus its market signals and deep-data context)
to a free-tier LLM with the airtight system prompt in system_prompt.md, and
stores the plain-English briefing on the prediction row.

Free inference options (pick via LLM_PROVIDER env var):
  * groq   — Groq free tier, llama-3.3-70b-versatile (fast, generous limits)
  * gemini — Google AI Studio free tier, gemini-2.0-flash
  * ollama — fully local, e.g. `ollama run llama3.1:8b` (zero cloud dependency)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from src.betting.market_dynamics import detect_anomalies
from src.utils.db import get_conn, query
from src.utils.odds_math import decimal_to_american

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")


def call_llm(system: str, user: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "groq")
    if provider == "groq":
        import time
        for attempt in range(4):
            r = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "temperature": 0.3,
                    "max_tokens": 400,
                },
                timeout=60,
            )
            if r.status_code == 429 and attempt < 3:  # free-tier rate limit: back off
                wait = float(r.headers.get("retry-after", 20)) + 2
                time.sleep(min(wait, 60))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    if provider == "gemini":
        r = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={os.environ['GEMINI_API_KEY']}",
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 400},
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    # ollama (local)
    r = httpx.post(
        "http://localhost:11434/api/chat",
        json={"model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}],
              "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


METRIC_LABEL = {
    "net_rating": "net rating", "pf_pg": "points/game", "pa_pg": "points allowed/game",
    "net_pg": "scoring margin", "win_pct": "win rate", "epa_off": "offensive EPA/play",
    "epa_def": "defensive EPA/play", "ops": "OPS", "runs_pg": "runs/game", "era": "ERA",
    "whip": "WHIP", "gf_per_game": "goals/game", "ga_per_game": "goals allowed/game",
    "xg_for": "xG for", "xg_against": "xG against",
}


def template_briefing(d: dict) -> str:
    """Deterministic briefing straight from the dossier — used whenever the LLM
    is unavailable (quota/keys) so no card is ever left blank."""
    p = d["pick"]
    prob = p["model_prob"]
    bullets = []
    hm, am = d["team_metrics"].get("home") or {}, d["team_metrics"].get("away") or {}
    for k in list(METRIC_LABEL):
        if k in hm and k in am and isinstance(hm[k], (int, float)) and isinstance(am[k], (int, float)):
            if abs(hm[k] - am[k]) > 1e-9:
                lead = "Home" if hm[k] > am[k] else "Away"
                if k in ("pa_pg", "era", "whip", "ga_per_game", "xg_against", "epa_def"):
                    lead = "Home" if hm[k] < am[k] else "Away"  # lower is better
                bullets.append(f"- {lead} side has the better {METRIC_LABEL[k]} "
                               f"({hm[k]:.2f} vs {am[k]:.2f}).")
    bullets = bullets[:3]
    rest = d.get("rest") or {}
    if rest.get("home_days") is not None and rest.get("away_days") is not None \
            and rest["home_days"] != rest["away_days"]:
        fresher = "home" if rest["home_days"] > rest["away_days"] else "away"
        bullets.append(f"- Rest edge: the {fresher} side has "
                       f"{max(rest['home_days'], rest['away_days'])} days vs "
                       f"{min(rest['home_days'], rest['away_days'])}.")
    w = d.get("weather") or {}
    if (w.get("wind_kph") or 0) >= 25:
        bullets.append(f"- Weather factor: {w['wind_kph']:.0f} km/h wind at the venue.")
    inj = d.get("injuries") or []
    if inj:
        bullets.append(f"- {len(inj)} recent injury reports across both rosters.")
    if not bullets:
        bullets = ["- Probabilities come from each team's season scoring rates."]
    if p["is_value_bet"]:
        head = (f"🎯 **THE PICK:** {p['outcome']} ({p['market']})\n\n"
                f"📊 **CALCULATED EDGE:** +{p['ev_pct']}% EV — model has this at "
                f"{prob:.0%}.\n\n💰 **STAKE:** ${p.get('stake', 0):,.2f} "
                f"({p['kelly_frac']:.1%} of bankroll, quarter-Kelly)")
        risk = ("⚠️ **RISK CHECK:** A favorite at these numbers still loses often; "
                "this is a probabilistic edge, not a certainty.")
    else:
        head = (f"🧠 **MODEL LEAN (NO BET)**\n\n🎯 {p['outcome']} — model {prob:.0%}.\n\n"
                f"💰 **STAKE:** $0 — no betting edge at current prices")
        risk = ("⚠️ **RISK CHECK:** The market already reflects the model's view of "
                "this game, so there is no profitable bet even if the lean wins.")
    return head + "\n\n⚔️ **TACTICAL EDGE:**\n" + "\n".join(bullets) + "\n\n" + risk


def build_dossier(conn, pred: dict) -> dict:
    """Assemble everything The Analyst is allowed to reason from."""
    game = query(conn, """
        select g.*, gc.weather, gc.sentiment, gc.tz_crossed_away
        from games g left join game_context gc using (event_id)
        where g.event_id = %s""", (pred["event_id"],))[0]
    hm = query(conn, """select metrics from team_metrics where team_id=%s
                        order by as_of desc limit 1""", (game["home_team_id"],))
    am = query(conn, """select metrics from team_metrics where team_id=%s
                        order by as_of desc limit 1""", (game["away_team_id"],))
    injuries = query(conn, """
        select p.name, p.team_id, i.status, left(i.detail, 120) detail
        from injuries i join players p using (player_id)
        where p.team_id in (%s, %s) and i.status in ('out','doubtful','questionable')
          and i.reported_at > now() - interval '3 days'
        order by i.reported_at desc limit 10
    """, (game["home_team_id"], game["away_team_id"]))
    signals = [s.__dict__ for s in detect_anomalies(conn, pred["event_id"])]

    market_prob = round(1 / pred["best_decimal"], 4) if pred["best_decimal"] else None
    return {
        # rsplit: sport keys contain underscores (soccer_epl_Arsenal), team
        # display names never do — split('_',1) would leave "epl_Arsenal"
        "matchup": f"{game['away_team_id'].rsplit('_', 1)[-1]} @ {game['home_team_id'].rsplit('_', 1)[-1]}",
        "commence_time": str(game["commence_time"]),
        "pick": {
            "outcome": pred["outcome"], "market": pred["market"],
            "line": pred["line"],
            "american_odds": decimal_to_american(pred["best_decimal"]),
            "bookmaker": pred["best_book"],
            "model_prob": pred["model_prob"], "market_prob": market_prob,
            "ev_pct": pred["ev_pct"], "kelly_frac": pred["kelly_frac"],
            "stake": pred.get("stake"),
            "is_value_bet": pred["is_value_bet"],
        },
        "team_metrics": {"home": hm[0]["metrics"] if hm else None,
                         "away": am[0]["metrics"] if am else None},
        "rest": {"home_days": game["home_rest_days"], "away_days": game["away_rest_days"]},
        "travel_km": {"home": game["home_travel_km"], "away": game["away_travel_km"]},
        "weather": game["weather"],
        "sentiment": game["sentiment"],
        "injuries": injuries,
        "market_signals": signals,
    }


def main(max_bets: int = 40) -> None:
    """Brief one outcome per upcoming game: the value bet when there is one,
    otherwise the model's most likely outcome (a 'lean'), so the dashboard has
    reasoning for every card."""
    with get_conn() as conn:
        preds = query(conn, """
            select distinct on (p.event_id) p.*
            from predictions p
            join games g using (event_id)
            where p.reasoning is null and g.commence_time > now()
            order by p.event_id, p.is_value_bet desc, p.model_prob desc
            limit %s
        """, (max_bets,))
        bank = query(conn, "select balance from bankroll order by ts desc limit 1")
        balance = bank[0]["balance"] if bank else 1000.0
        llm_down = False
        for p in preds:
            p["stake"] = round(balance * (p["kelly_frac"] or 0), 2)
            try:
                dossier = build_dossier(conn, p)
            except Exception as e:
                print(f"[llm] pred {p['pred_id']} dossier failed: {e}")
                continue
            briefing, source = None, "llm"
            if not llm_down:
                try:
                    briefing = call_llm(SYSTEM_PROMPT, json.dumps(dossier, default=str))
                except Exception as e:
                    print(f"[llm] provider unavailable ({e}); switching to quick-read briefings")
                    llm_down = True
            if briefing is None:  # quota exhausted or provider down: stats-based fallback
                briefing, source = template_briefing(dossier), "template"
            with conn.cursor() as cur:
                cur.execute("update predictions set reasoning=%s where pred_id=%s",
                            (briefing, p["pred_id"]))
            print(f"[llm] analyzed pred {p['pred_id']} ({p['outcome']}) via {source}")


if __name__ == "__main__":
    main()

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
        "matchup": f"{game['away_team_id'].split('_',1)[1]} @ {game['home_team_id'].split('_',1)[1]}",
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


def main(max_bets: int = 15) -> None:
    with get_conn() as conn:
        preds = query(conn, """
            select p.* from predictions p
            join games g using (event_id)
            where p.is_value_bet and p.reasoning is null
              and g.commence_time > now()
            order by p.ev_pct desc limit %s
        """, (max_bets,))
        bank = query(conn, "select balance from bankroll order by ts desc limit 1")
        balance = bank[0]["balance"] if bank else 1000.0
        for p in preds:
            p["stake"] = round(balance * (p["kelly_frac"] or 0), 2)
            try:
                dossier = build_dossier(conn, p)
                briefing = call_llm(SYSTEM_PROMPT, json.dumps(dossier, default=str))
                with conn.cursor() as cur:
                    cur.execute("update predictions set reasoning=%s where pred_id=%s",
                                (briefing, p["pred_id"]))
                print(f"[llm] analyzed pred {p['pred_id']} ({p['outcome']} {p['ev_pct']}%)")
            except Exception as e:
                print(f"[llm] pred {p['pred_id']} failed: {e}")


if __name__ == "__main__":
    main()

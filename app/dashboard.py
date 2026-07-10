"""VegasEdge — model-first prediction board (Streamlit Community Cloud).

Dark trading-desk design. One flip-card per game (deduped by matchup, in
kickoff order, with LIVE state): the FRONT shows what OUR model predicts,
the BACK (click) shows the AI analyst's reasoning. A results ledger shows
every graded pick vs the real final score. Sportsbook prices stay backend.

Falls back to demo data with no DATABASE_URL:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import html as html_lib
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root importable
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

st.set_page_config(page_title="VegasEdge", page_icon="📈", layout="wide")

# ---- design tokens (CVD + contrast validated on #0d1117, dark band) --------
SURFACE = "#0d1117"
PANEL = "#141a23"
BORDER = "#222b38"
INK = "#e8eef5"
INK_MUTED = "#8b98a9"
GREEN = "#059669"
BLUE = "#0284c7"
AMBER = "#d97706"
VIOLET = "#9333ea"
ACCENT = "#00e68a"
LOSS = "#f85149"

GAME_HOURS = {"mlb": 3.3, "nba": 2.6, "nfl": 3.4, "nhl": 2.9}  # soccer default 2.3

st.markdown(f"""
<style>
  .stApp {{ background:
      radial-gradient(1100px 460px at 12% -8%, #122338 0%, transparent 60%),
      radial-gradient(900px 420px at 95% -12%, #1b1430 0%, transparent 55%),
      {SURFACE}; }}
  html, body, [class*="css"] {{ color: {INK}; font-family: 'Segoe UI', system-ui, sans-serif; }}
  h1, h2, h3 {{ color: {INK} !important; letter-spacing: -0.02em; }}
  .ve-kicker {{ font-size: 0.7rem; font-weight: 800; letter-spacing: 0.18em;
                text-transform: uppercase; margin-bottom: 2px; }}
  .ve-tile {{
    background: linear-gradient(180deg, {PANEL} 0%, #10151d 100%);
    border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 20px; height: 100%;
    border-top: 2px solid var(--tile-accent, {BORDER});
  }}
  .ve-tile .label {{ color: {INK_MUTED}; font-size: 0.68rem; text-transform: uppercase;
                     letter-spacing: 0.1em; margin-bottom: 6px; }}
  .ve-tile .value {{ color: {INK}; font-size: 1.65rem; font-weight: 750;
                     font-variant-numeric: tabular-nums; }}
  .ve-tile .sub {{ color: {INK_MUTED}; font-size: 0.78rem; margin-top: 2px; }}
  .ve-tile .up {{ color: {ACCENT}; }} .ve-tile .down {{ color: {LOSS}; }}
  .ve-res {{
    display: grid; grid-template-columns: 54px 1.6fr 1.4fr 0.9fr 0.8fr; gap: 12px;
    align-items: center; background: {PANEL}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
  }}
  .ve-res .chipW, .ve-res .chipL, .ve-res .chipP {{
    font-weight: 800; font-size: 0.78rem; text-align: center; border-radius: 8px;
    padding: 6px 0; letter-spacing: 0.05em;
  }}
  .ve-res .chipW {{ color: #05130c; background: {ACCENT}; }}
  .ve-res .chipL {{ color: #fff; background: {LOSS}; }}
  .ve-res .chipP {{ color: {INK_MUTED}; background: #222b38; }}
  .ve-res .m {{ font-weight: 700; }} .ve-res .sub {{ color: {INK_MUTED}; font-size: 0.75rem; }}
  .ve-res .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data layer

@st.cache_data(ttl=180)
def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict | None, bool]:
    """(upcoming picks, bankroll curve, results ledger, all-time record, is_demo)."""
    try:
        secret_url = st.secrets.get("DATABASE_URL", "")
    except Exception:
        secret_url = ""
    if os.environ.get("DATABASE_URL") or secret_url:
        os.environ.setdefault("DATABASE_URL", secret_url)
        from src.utils.db import get_conn, query
        with get_conn() as conn:
            picks = pd.DataFrame(query(conn, """
                select distinct on (p.event_id, p.outcome)
                       p.event_id, p.outcome, p.model_prob, p.ev_pct, p.kelly_frac,
                       p.is_value_bet, p.reasoning, p.created_at,
                       g.sport, g.commence_time, g.home_team_id, g.away_team_id
                from predictions p join games g using (event_id)
                where p.market = 'h2h' and g.status = 'scheduled'
                  and g.commence_time > now() - interval '6 hours'
                  and g.commence_time < now() + interval '60 hours'
                order by p.event_id, p.outcome, p.created_at desc
                limit 500
            """))
            bank = pd.DataFrame(query(conn, "select ts, balance from bankroll order by ts"))
            results = pd.DataFrame(query(conn, """
                with latest as (
                  select distinct on (p.event_id, p.outcome)
                         p.event_id, p.outcome, p.model_prob, p.is_value_bet, p.pred_id
                  from predictions p where p.market = 'h2h'
                  order by p.event_id, p.outcome, p.created_at desc
                ), tops as (
                  select distinct on (event_id) event_id, outcome pick, model_prob, pred_id
                  from latest order by event_id, model_prob desc
                )
                select t.pick, t.model_prob, g.sport, g.commence_time,
                       g.home_team_id, g.away_team_id, g.home_score, g.away_score,
                       br.result bet_result, br.pnl, br.stake
                from tops t
                join games g using (event_id)
                left join (
                  -- match bets by event, not pred_id: repricing inserts a newer
                  -- prediction row, so the graded bet hangs off an older pred_id
                  -- and a pred_id join silently drops it from the ledger
                  select p2.event_id,
                         case when sum(br0.pnl) >= 0 then 'win' else 'loss' end result,
                         sum(br0.pnl) pnl, sum(br0.stake) stake
                  from bet_results br0 join predictions p2 using (pred_id)
                  group by p2.event_id
                ) br on br.event_id = t.event_id
                where g.status = 'final'
                order by g.commence_time desc limit 30
            """))
            # All-time record over EVERY graded pick — the ledger above is
            # capped at 30 rows for display, so counting it would freeze the
            # record at a rolling 30-game window.
            record = query(conn, """
                with latest as (
                  select distinct on (p.event_id, p.outcome)
                         p.event_id, p.outcome, p.model_prob, p.pred_id
                  from predictions p where p.market = 'h2h'
                  order by p.event_id, p.outcome, p.created_at desc
                ), tops as (
                  select distinct on (event_id) event_id, outcome pick, pred_id
                  from latest order by event_id, model_prob desc
                ), graded as (
                  select t.pick,
                         case when g.home_score > g.away_score
                                then substring(g.home_team_id from length(g.sport) + 2)
                              when g.away_score > g.home_score
                                then substring(g.away_team_id from length(g.sport) + 2)
                              else 'Draw' end as winner
                  from tops t
                  join games g using (event_id)
                  where g.status = 'final'
                    and g.home_score is not null and g.away_score is not null
                )
                select count(*) filter (where pick = winner)  as wins,
                       count(*) filter (where pick <> winner) as losses,
                       (select coalesce(sum(pnl), 0) from bet_results) as pnl
                from graded
            """)[0]
        return picks, bank, results, record, False
    return _demo_picks(), _demo_bankroll(), _demo_results(), None, True


def _demo_picks() -> pd.DataFrame:
    now = datetime.now()
    def row(eid, out, prob, ev, kf, val, why, sport, hrs, home, away):
        return dict(event_id=eid, outcome=out, model_prob=prob, ev_pct=ev,
                    kelly_frac=kf, is_value_bet=val, reasoning=why, sport=sport,
                    created_at=now, commence_time=now + timedelta(hours=hrs),
                    home_team_id=home, away_team_id=away)
    demo_why = ("🎯 **THE PICK:** Milwaukee Brewers ML\n\n📊 **CALCULATED EDGE:** +14.5% EV\n\n"
                "💰 **STAKE:** $50 (5.0%, quarter-Kelly)\n\n⚔️ **TACTICAL EDGE:**\n"
                "- Brewers 5.1 runs/game vs 4.2.\n- Staff ERA 3.32 vs 4.12.\n\n"
                "⚠️ **RISK CHECK:** Baseball is high variance.")
    return pd.DataFrame([
        row("d1", "Argentina", 0.63, -2.1, 0, False, "🧠 **MODEL LEAN (NO BET)**\n\nArgentina controls play.",
            "worldcup", -0.9, "worldcup_Argentina", "worldcup_Egypt"),
        row("d1", "Draw", 0.22, -8, 0, False, None, "worldcup", -0.9, "worldcup_Argentina", "worldcup_Egypt"),
        row("d1", "Egypt", 0.15, -11, 0, False, None, "worldcup", -0.9, "worldcup_Argentina", "worldcup_Egypt"),
        row("d2", "Milwaukee Brewers", 0.75, 14.5, 0.05, True, demo_why,
            "mlb", 3.0, "mlb_Milwaukee Brewers", "mlb_St. Louis Cardinals"),
        row("d2", "St. Louis Cardinals", 0.25, -18, 0, False, None,
            "mlb", 3.0, "mlb_Milwaukee Brewers", "mlb_St. Louis Cardinals"),
    ])


def _demo_results() -> pd.DataFrame:
    now = datetime.now()
    return pd.DataFrame([
        dict(pick="Boston Celtics", model_prob=0.64, sport="nba",
             commence_time=now - timedelta(days=1), home_team_id="nba_Boston Celtics",
             away_team_id="nba_Miami Heat", home_score=112, away_score=101,
             bet_result="win", pnl=39.0, stake=50.0),
        dict(pick="Arsenal", model_prob=0.51, sport="soccer_epl",
             commence_time=now - timedelta(days=2), home_team_id="soccer_epl_Arsenal",
             away_team_id="soccer_epl_Chelsea", home_score=1, away_score=1,
             bet_result="loss", pnl=-17.0, stake=17.0),
        dict(pick="New York Yankees", model_prob=0.58, sport="mlb",
             commence_time=now - timedelta(days=2), home_team_id="mlb_New York Yankees",
             away_team_id="mlb_Tampa Bay Rays", home_score=6, away_score=2,
             bet_result=None, pnl=None, stake=None),
    ])


def _demo_bankroll() -> pd.DataFrame:
    import math
    days = pd.date_range(end=datetime.now(), periods=45)
    bal, curve = 1000.0, []
    for i, d in enumerate(days):
        bal *= 1 + 0.004 * math.sin(i * 1.7) + 0.0022
        curve.append({"ts": d, "balance": round(bal, 2)})
    return pd.DataFrame(curve)


def md_to_html(md: str) -> str:
    if not md:
        return ""
    md = re.sub(r"@\s*[+-]?\d+(\.\d+)?\s*\([a-zA-Z0-9_ ]+\)", "", md)  # strip book prices
    s = html_lib.escape(md)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = s.replace("\n- ", "\n• ").replace("\n", "<br>")
    return s


LEAGUE_LABEL = {
    "mlb": "MLB", "nba": "NBA", "nfl": "NFL", "nhl": "NHL", "worldcup": "WORLD CUP",
    "soccer_epl": "PREMIER LEAGUE", "soccer_laliga": "LA LIGA", "soccer_seriea": "SERIE A",
    "soccer_bundesliga": "BUNDESLIGA", "soccer_ucl": "CHAMPIONS LEAGUE", "soccer_mls": "MLS",
}


def build_games(picks: pd.DataFrame, bankroll: float) -> list[dict]:
    if picks.empty:
        return []
    games: dict[tuple, dict] = {}
    for event_id, grp in picks.groupby("event_id", sort=False):
        g0 = grp.iloc[0]
        home = str(g0["home_team_id"]).split("_")[-1]
        away = str(g0["away_team_id"]).split("_")[-1]
        kick = pd.Timestamp(g0["commence_time"])
        # dedupe by matchup+day, not event id (the same fixture can arrive under
        # different ids from the odds feed vs backfill) — keep newest predictions
        key = (str(g0["sport"]), home, away, kick.date().isoformat())
        newest = grp["created_at"].max()
        # rank duplicates: a listing whose game window hasn't passed beats a stale
        # "zombie" listing (wrong old kickoff); then prefer newest predictions
        dur = GAME_HOURS.get(str(g0["sport"]), 2.3)
        not_ended = datetime.now() < (kick.tz_localize(None) + timedelta(hours=dur))
        rank = (not_ended, newest)
        if key in games and games[key]["_rank"] >= rank:
            continue
        outs = []
        for _, r in grp.iterrows():
            outs.append({"name": r["outcome"],
                         "prob": round(float(r["model_prob"] or 0), 4),
                         "value": bool(r["is_value_bet"]),
                         "stake": round(bankroll * float(r["kelly_frac"] or 0), 2),
                         "edge": round(float(r["ev_pct"] or 0), 1)})
        order = {home: 0, "Draw": 1, away: 2}
        outs.sort(key=lambda o: order.get(o["name"], 3))
        top = max(outs, key=lambda o: o["prob"])
        pick = next((o for o in outs if o["value"]), None)
        reasoning = next((r for r in grp["reasoning"]
                          if isinstance(r, str) and r.strip()), "")
        sport = str(g0["sport"])
        games[key] = {
            "_rank": rank,
            "id": str(event_id),
            "league": LEAGUE_LABEL.get(sport, sport.upper()),
            "kick": kick.isoformat(),
            "durH": GAME_HOURS.get(sport, 2.3),
            "home": home, "away": away,
            "outcomes": outs,
            "top": top["name"], "topProb": top["prob"],
            "isBet": pick is not None,
            "betName": pick["name"] if pick else None,
            "stake": pick["stake"] if pick else 0,
            "edge": pick["edge"] if pick else None,
            "why": md_to_html(reasoning) or
                   "<i>Analysis being generated — the AI analyst runs hourly.</i>",
        }
    out = sorted(games.values(), key=lambda g: g["kick"])  # first kickoff on top
    for g in out:
        g.pop("_rank", None)
    return out


picks, bank, results, record, is_demo = load_frames()
balance = float(bank["balance"].iloc[-1]) if not bank.empty else 1000.0
games = build_games(picks, balance)

# all-time record comes from the aggregate query (the ledger is only the
# newest 30 games); demo mode has no aggregate, so count the demo rows
rec_w = rec_l = 0
pnl_total = 0.0
graded_rows = []
if not results.empty:
    for _, r in results.iterrows():
        hs, as_ = r["home_score"], r["away_score"]
        if hs is None or as_ is None:
            continue
        home = str(r["home_team_id"]).split("_")[-1]
        away = str(r["away_team_id"]).split("_")[-1]
        winner = home if hs > as_ else away if as_ > hs else "Draw"
        correct = (r["pick"] == winner)
        rec_w += int(correct); rec_l += int(not correct)
        # SQL NULLs surface as NaN here, and NaN is truthy — `or 0` won't catch it.
        pnl = float(r["pnl"]) if pd.notna(r["pnl"]) else 0.0
        pnl_total += pnl
        # US local date, not UTC: a 7pm West Coast game lands on the next UTC
        # day and would collide with the following game of the series.
        when = pd.Timestamp(r["commence_time"]).tz_convert("America/New_York")
        graded_rows.append({
            "correct": correct, "league": LEAGUE_LABEL.get(str(r["sport"]), str(r["sport"]).upper()),
            "match": f"{away} @ {home}", "score": f"{int(as_)}–{int(hs)}",
            "pick": r["pick"], "prob": float(r["model_prob"] or 0),
            "was_bet": pd.notna(r["bet_result"]), "pnl": pnl,
            "when": when.strftime("%b %d"),
        })
if record is not None:
    rec_w, rec_l = int(record["wins"]), int(record["losses"])
    pnl_total = float(record["pnl"])

# ----------------------------------------------------------------- header row

st.markdown(
    f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px;'>"
    f"<span style='font-size:2.05rem;font-weight:800;letter-spacing:-0.03em;"
    f"background:linear-gradient(90deg,{ACCENT},#22d3ee 60%,{VIOLET});"
    f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>📈 VegasEdge</span>"
    f"<span style='color:{ACCENT};font-weight:700;font-size:0.78rem;letter-spacing:0.16em;'>MODEL BOARD</span>"
    f"<span style='color:{INK_MUTED};font-size:0.85rem;'>{datetime.now():%A, %B %d}</span></div>",
    unsafe_allow_html=True)
if is_demo:
    st.caption("⚠️ Demo data — set DATABASE_URL to connect your live database.")

pnl_30 = balance - (float(bank["balance"].iloc[max(0, len(bank) - 30)]) if not bank.empty else balance)
n_bets = sum(1 for g in games if g["isBet"])
live_now = sum(1 for g in games
               if 0 <= (datetime.now() - pd.Timestamp(g["kick"]).tz_localize(None)
                        ).total_seconds() / 3600 <= g["durH"])

c1, c2, c3, c4 = st.columns(4)
for col, accent, label, value, sub, cls in (
    (c1, ACCENT, "Bankroll", f"${balance:,.2f}", f"{pnl_30:+,.2f} last 30 days",
     "up" if pnl_30 >= 0 else "down"),
    (c2, "#22d3ee", "Model Record", f"{rec_w}–{rec_l}" if rec_w + rec_l else "—",
     "graded picks, all games", "up" if rec_w >= rec_l else "down"),
    (c3, VIOLET, "Picks Today", f"{n_bets}",
     f"{live_now} live now · {len(games)} on the board", "up"),
    (c4, AMBER, "Staking", "¼ Kelly", "hard cap 5% per bet", ""),
):
    col.markdown(f"<div class='ve-tile' style='--tile-accent:{accent}'>"
                 f"<div class='label'>{label}</div><div class='value'>{value}</div>"
                 f"<div class='sub {cls}'>{sub}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ the board

BOARD_TEMPLATE = r"""
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; font-family: 'Segoe UI', system-ui, sans-serif; }
  .wrap { padding: 4px 2px 30px; }
  .toolbar { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; }
  .search { flex: 1; background: #141a23; border: 1px solid #222b38; color: #e8eef5;
            border-radius: 12px; padding: 12px 18px; font-size: 0.95rem; outline: none;
            transition: border 0.2s, box-shadow 0.2s; }
  .search:focus { border-color: #22d3ee; box-shadow: 0 0 0 3px rgba(34,211,238,0.15); }
  .hint { color: #8b98a9; font-size: 0.8rem; white-space: nowrap; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 16px; }
  .flip { perspective: 1400px; height: 392px; cursor: pointer; }
  .inner { position: relative; width: 100%; height: 100%;
           transition: transform 0.65s cubic-bezier(0.4,0.1,0.2,1); transform-style: preserve-3d; }
  .flip.flipped .inner { transform: rotateY(180deg); }
  .face { position: absolute; inset: 0; backface-visibility: hidden; border-radius: 18px;
          border: 1px solid #232d3b; overflow: hidden;
          background: linear-gradient(165deg, #18222e 0%, #10161f 72%);
          box-shadow: 0 8px 26px rgba(0,0,0,0.35); }
  .flip:hover .face { border-color: #31405a; box-shadow: 0 14px 36px rgba(0,0,0,0.5); }
  .flip.bet .front { border-color: rgba(0,230,138,0.45);
      background: linear-gradient(165deg, #14251f 0%, #10161f 70%); }
  .flip.live .front::after { content: ""; position: absolute; inset: 0; pointer-events: none;
      background: radial-gradient(420px 120px at 85% 0%, rgba(248,81,73,0.14), transparent 70%); }
  .front { padding: 18px 20px 14px; display: flex; flex-direction: column; }
  .back { transform: rotateY(180deg); padding: 18px 22px; display: flex; flex-direction: column;
          background: linear-gradient(165deg, #131c28 0%, #0f141c 70%); }
  .rowtop { display: flex; justify-content: space-between; align-items: center; }
  .chip { font-size: 0.64rem; font-weight: 800; letter-spacing: 0.12em; padding: 4px 10px;
          border-radius: 999px; border: 1px solid transparent; }
  .status { font-size: 0.72rem; font-weight: 700; color: #8b98a9;
            font-variant-numeric: tabular-nums; display: flex; align-items: center; gap: 6px; }
  .status.live { color: #f85149; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: #f85149;
         animation: blink 1.1s ease-in-out infinite; }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }
  .duel { display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; align-items: center;
          margin: 16px 0 6px; }
  .side { display: flex; flex-direction: column; align-items: center; gap: 7px; min-width: 0; }
  .ava { width: 52px; height: 52px; border-radius: 16px; display: flex; align-items: center;
         justify-content: center; font-weight: 800; font-size: 1.02rem; color: #fff;
         letter-spacing: 0.02em; box-shadow: inset 0 -10px 18px rgba(0,0,0,0.3),
         0 4px 12px rgba(0,0,0,0.35); }
  .side .nm { color: #c7d3e2; font-size: 0.78rem; font-weight: 650; max-width: 100%;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .side .pb { font-size: 1.05rem; font-weight: 800; color: #e8eef5;
              font-variant-numeric: tabular-nums; }
  .side.win .pb { color: #00e68a; }
  .vs { color: #55647a; font-size: 0.8rem; font-weight: 700; }
  .verdict { display: flex; align-items: center; justify-content: center; gap: 10px;
             margin: 10px 0 12px; flex-wrap: wrap; }
  .badge { font-size: 0.66rem; font-weight: 800; letter-spacing: 0.1em; padding: 5px 12px;
           border-radius: 999px; }
  .badge.bet { color: #00e68a; background: rgba(0,230,138,0.13);
               border: 1px solid rgba(0,230,138,0.6); animation: pulse 1.8s infinite; }
  .badge.lean { color: #9fb1c4; background: rgba(139,152,169,0.1); border: 1px solid #2a3442; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(0,230,138,0.3); }
                     50% { box-shadow: 0 0 14px 2px rgba(0,230,138,0.22); } }
  .verdict .who { font-size: 0.95rem; font-weight: 750; color: #e8eef5; }
  .verdict .stake { color: #00e68a; font-weight: 750; font-size: 0.88rem; }
  .bars { display: flex; flex-direction: column; gap: 8px; margin-top: auto; }
  .barrow { display: grid; grid-template-columns: 108px 1fr 48px; gap: 10px; align-items: center; }
  .barrow .name { color: #aebbcb; font-size: 0.78rem; white-space: nowrap; overflow: hidden;
                  text-overflow: ellipsis; }
  .barrow .name.win { color: #e8eef5; font-weight: 700; }
  .track { height: 8px; background: #1b2430; border-radius: 999px; overflow: hidden; }
  .fill { height: 100%; border-radius: 999px; transition: width 0.9s cubic-bezier(0.2,0.8,0.2,1); }
  .barrow .pct { color: #e8eef5; font-size: 0.8rem; font-weight: 650; text-align: right;
                 font-variant-numeric: tabular-nums; }
  .tap { margin-top: 12px; color: #55647a; font-size: 0.7rem; letter-spacing: 0.06em;
         text-align: center; }
  .back h4 { color: #00e68a; font-size: 0.7rem; letter-spacing: 0.16em; margin-bottom: 10px; }
  .why { color: #c6d2e0; font-size: 0.82rem; line-height: 1.55; overflow-y: auto; flex: 1;
         padding-right: 6px; }
  .why::-webkit-scrollbar { width: 6px; }
  .why::-webkit-scrollbar-thumb { background: #2a3442; border-radius: 3px; }
  .why b { color: #e8eef5; }
  .backfoot { margin-top: 10px; color: #55647a; font-size: 0.7rem; text-align: center; }
  .empty { color: #8b98a9; padding: 40px; text-align: center; border: 1px dashed #2a3442;
           border-radius: 16px; }
</style>
<div class="wrap">
  <div class="toolbar">
    <input id="q" class="search" placeholder="🔍  Search any league, team or match…">
    <span class="hint">click a card for the model's reasoning</span>
  </div>
  <div id="grid" class="grid"></div>
  <div id="empty" class="empty" style="display:none">No games match your search.</div>
</div>
<script>
const GAMES = __GAMES_JSON__;
const BAR = ["#0284c7", "#d97706", "#9333ea"];        // home, draw, away
const AVA = [["#0e7490","#0369a1"],["#047857","#059669"],["#b45309","#d97706"],
             ["#7e22ce","#9333ea"],["#be123c","#e11d48"],["#1d4ed8","#3b82f6"]];
const CHIP = {"MLB":"#38bdf8","NBA":"#fb923c","NFL":"#4ade80","NHL":"#a5b4fc",
              "WORLD CUP":"#00e68a","PREMIER LEAGUE":"#c084fc","LA LIGA":"#fbbf24",
              "SERIE A":"#38bdf8","BUNDESLIGA":"#f87171","CHAMPIONS LEAGUE":"#818cf8","MLS":"#2dd4bf"};
const grid = document.getElementById("grid");

const initials = n => n.split(/\s+/).filter(w => /^[A-Za-z]/.test(w))
    .map(w => w[0]).slice(0, 2).join("").toUpperCase() || n.slice(0, 2).toUpperCase();
const hue = n => { let h = 0; for (const c of n) h = (h * 31 + c.charCodeAt(0)) % AVA.length; return AVA[h]; };

function status(g) {
  const now = new Date(), k = new Date(g.kick);
  const hrs = (now - k) / 36e5;
  if (hrs >= 0 && hrs <= g.durH) return {cls: "live", html: `<span class="dot"></span>LIVE`};
  if (hrs > g.durH) return {cls: "", html: "ENDED · grading soon"};
  const mins = Math.round(-hrs * 60);
  const t = k.toLocaleTimeString([], {hour: "numeric", minute: "2-digit"});
  if (mins < 60) return {cls: "", html: `in ${mins}m · ${t}`};
  if (k.toDateString() === now.toDateString()) return {cls: "", html: `Today · ${t}`};
  return {cls: "", html: k.toLocaleDateString([], {weekday: "short"}) + " · " + t};
}

function card(g) {
  const s = status(g);
  const el = document.createElement("div");
  el.className = "flip" + (g.isBet ? " bet" : "") + (s.cls === "live" ? " live" : "");
  el.dataset.text = (g.league + " " + g.home + " " + g.away).toLowerCase();
  const [c1, c2] = [hue(g.away), hue(g.home)];
  const aProb = g.outcomes.find(o => o.name === g.away), hProb = g.outcomes.find(o => o.name === g.home);
  const chipC = CHIP[g.league] || "#9fb1c4";
  const bars = g.outcomes.map((o, i) => `
    <div class="barrow">
      <div class="name ${o.name === g.top ? "win" : ""}">${o.name === g.top ? "✓ " : ""}${o.name}</div>
      <div class="track"><div class="fill" style="width:0%;background:${BAR[i % 3]}" data-w="${(o.prob*100).toFixed(1)}"></div></div>
      <div class="pct">${(o.prob*100).toFixed(0)}%</div>
    </div>`).join("");
  const verdict = g.isBet
    ? `<span class="badge bet">🔥 MODEL PICK</span><span class="who">${g.betName}</span>
       <span class="stake">bet $${g.stake.toFixed(0)}</span>`
    : `<span class="badge lean">LEAN · NO BET</span><span class="who">${g.top} ${(g.topProb*100).toFixed(0)}%</span>`;
  el.innerHTML = `
    <div class="inner">
      <div class="face front">
        <div class="rowtop">
          <span class="chip" style="color:${chipC};background:${chipC}22;border-color:${chipC}55">${g.league}</span>
          <span class="status ${s.cls}">${s.html}</span>
        </div>
        <div class="duel">
          <div class="side ${g.top === g.away ? "win" : ""}">
            <div class="ava" style="background:linear-gradient(140deg,${c1[0]},${c1[1]})">${initials(g.away)}</div>
            <div class="nm">${g.away}</div><div class="pb">${aProb ? (aProb.prob*100).toFixed(0) + "%" : ""}</div>
          </div>
          <div class="vs">@</div>
          <div class="side ${g.top === g.home ? "win" : ""}">
            <div class="ava" style="background:linear-gradient(140deg,${c2[0]},${c2[1]})">${initials(g.home)}</div>
            <div class="nm">${g.home}</div><div class="pb">${hProb ? (hProb.prob*100).toFixed(0) + "%" : ""}</div>
          </div>
        </div>
        <div class="verdict">${verdict}</div>
        <div class="bars">${bars}</div>
        <div class="tap">TAP FOR THE WHY ⟲</div>
      </div>
      <div class="face back">
        <h4>WHY THE MODEL SAYS ${g.top.toUpperCase()}</h4>
        <div class="why">${g.why}</div>
        <div class="backfoot">⟲ tap to flip back</div>
      </div>
    </div>`;
  el.addEventListener("click", () => el.classList.toggle("flipped"));
  return el;
}

GAMES.forEach(g => grid.appendChild(card(g)));
requestAnimationFrame(() => setTimeout(() =>
  document.querySelectorAll(".fill").forEach(f => f.style.width = f.dataset.w + "%"), 60));

document.getElementById("q").addEventListener("input", e => {
  const q = e.target.value.toLowerCase().trim();
  let shown = 0;
  document.querySelectorAll(".flip").forEach(c => {
    const hit = !q || c.dataset.text.includes(q);
    c.style.display = hit ? "" : "none"; if (hit) shown++;
  });
  document.getElementById("empty").style.display = shown ? "none" : "";
});
</script>
"""

st.markdown(f"<div class='ve-kicker' style='color:{ACCENT}'>THE BOARD</div>", unsafe_allow_html=True)
st.markdown("### 🧠 Today's Games — in kickoff order")
if games:
    payload = json.dumps(games).replace("</", "<\\/")
    rows = -(-len(games) // 2)
    height = min(150 + rows * 412, 1750)
    components.html(BOARD_TEMPLATE.replace("__GAMES_JSON__", payload),
                    height=height, scrolling=True)
else:
    st.info("No upcoming games priced right now — the hourly pipeline will fill the board.")

# ------------------------------------------------------------- results ledger

st.markdown(f"<div class='ve-kicker' style='color:#22d3ee'>ACCOUNTABILITY</div>",
            unsafe_allow_html=True)
st.markdown("### 📋 Model vs Reality — every graded pick")
if graded_rows:
    wl = f"{rec_w}–{rec_l}"
    acc = rec_w / max(rec_w + rec_l, 1) * 100
    st.markdown(f"<div style='color:{INK_MUTED};margin-bottom:10px;'>Record "
                f"<b style='color:{INK}'>{wl}</b> ({acc:.0f}%) · Bet P/L "
                f"<b style='color:{ACCENT if pnl_total >= 0 else LOSS}'>{pnl_total:+,.2f}</b></div>",
                unsafe_allow_html=True)
    for r in graded_rows[:14]:
        chip = "chipW" if r["correct"] else "chipL"
        chip_txt = "WIN" if r["correct"] else "MISS"
        pnl_html = (f"<div class='num' style='color:{ACCENT if (r['pnl'] or 0) >= 0 else LOSS}'>"
                    f"{(r['pnl'] or 0):+,.0f}</div>" if r["was_bet"]
                    else f"<div class='num sub'>no bet</div>")
        st.markdown(
            f"<div class='ve-res'><div class='{chip}'>{chip_txt}</div>"
            f"<div><span class='m'>{r['match']}</span><br><span class='sub'>{r['league']} · {r['when']}</span></div>"
            f"<div>picked <span class='m'>{r['pick']}</span> <span class='sub'>({r['prob']:.0%})</span></div>"
            f"<div class='num m'>{r['score']}</div>{pnl_html}</div>",
            unsafe_allow_html=True)
else:
    st.markdown(f"<div class='ve-tile'><div class='label'>No graded games yet</div>"
                f"<div class='sub'>Results land here after tonight's games finish — the daily "
                f"pipeline grades every pick against the real final score, win or lose.</div></div>",
                unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------- bankroll section

col_chart, col_next = st.columns([3, 2])
with col_chart:
    st.markdown(f"<div class='ve-kicker' style='color:{GREEN}'>CAPITAL</div>", unsafe_allow_html=True)
    st.markdown("### 💰 Bankroll")
    if not bank.empty:
        fig = go.Figure(go.Scatter(
            x=bank["ts"], y=bank["balance"], mode="lines",
            line=dict(color=GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(5,150,105,0.10)",
            hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>"))
        fig.update_layout(template=None, height=250, margin=dict(l=52, r=8, t=8, b=8),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color=INK_MUTED, showgrid=False),
                          yaxis=dict(color=INK_MUTED, gridcolor=BORDER, tickprefix="$"),
                          hoverlabel=dict(bgcolor=PANEL, font_color=INK))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with col_next:
    st.markdown(f"<div class='ve-kicker' style='color:{VIOLET}'>ACTION</div>", unsafe_allow_html=True)
    st.markdown("### 🎯 Today's stakes")
    any_bet = False
    for g in games:
        if g["isBet"]:
            any_bet = True
            st.markdown(
                f"<div class='ve-tile' style='--tile-accent:{ACCENT};margin-bottom:10px;'>"
                f"<div class='label'>{g['league']} · {g['away']} @ {g['home']}</div>"
                f"<div class='value' style='font-size:1.12rem;'>{g['betName']} — bet ${g['stake']:,.0f}</div>"
                f"<div class='sub up'>model edge +{g['edge']}%</div></div>",
                unsafe_allow_html=True)
    if not any_bet:
        st.markdown(f"<div class='ve-tile'><div class='label'>No bets today</div>"
                    f"<div class='sub'>The model found no mispriced games. Not betting is a "
                    f"position — it's how the bankroll survives.</div></div>",
                    unsafe_allow_html=True)

st.markdown("---")
st.caption("VegasEdge is an analytical tool. Probabilities are model estimates; losses are "
           "expected and normal. Soccer results are for 90 minutes. Bet only where legal and "
           "only what you can afford to lose. 21+.")

"""VegasEdge — model-first prediction board (Streamlit Community Cloud).

Design language: dark trading-desk. One flip-card per game: the FRONT shows
what OUR model predicts (every game, every outcome, with probabilities); the
BACK (click to flip) shows the AI analyst's reasoning. Sportsbook prices stay
in the backend — the board only surfaces our numbers, the verdict, and stakes.

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
GREEN = "#059669"      # chart/bar series
BLUE = "#0284c7"
AMBER = "#d97706"
VIOLET = "#9333ea"
ACCENT = "#00e68a"     # UI accent (badges/glow only, never chart series)
LOSS = "#f85149"

st.markdown(f"""
<style>
  .stApp {{ background: radial-gradient(1200px 500px at 20% -10%, #101d2d 0%, {SURFACE} 55%); }}
  html, body, [class*="css"] {{ color: {INK}; font-family: 'Segoe UI', system-ui, sans-serif; }}
  h1, h2, h3 {{ color: {INK} !important; letter-spacing: -0.02em; }}
  .ve-tile {{
    background: linear-gradient(180deg, {PANEL} 0%, #10151d 100%);
    border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 22px; height: 100%;
  }}
  .ve-tile .label {{ color: {INK_MUTED}; font-size: 0.7rem; text-transform: uppercase;
                     letter-spacing: 0.1em; margin-bottom: 6px; }}
  .ve-tile .value {{ color: {INK}; font-size: 1.75rem; font-weight: 750;
                     font-variant-numeric: tabular-nums; }}
  .ve-tile .sub {{ color: {INK_MUTED}; font-size: 0.8rem; margin-top: 2px; }}
  .ve-tile .up {{ color: {ACCENT}; }} .ve-tile .down {{ color: {LOSS}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data layer

@st.cache_data(ttl=300)
def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    try:
        secret_url = st.secrets.get("DATABASE_URL", "")
    except Exception:
        secret_url = ""
    if os.environ.get("DATABASE_URL") or secret_url:
        os.environ.setdefault("DATABASE_URL", secret_url)
        from src.utils.db import get_conn, query
        with get_conn() as conn:
            picks = pd.DataFrame(query(conn, """
                select distinct on (p.event_id, p.market, p.outcome)
                       p.event_id, p.market, p.outcome, p.model_prob, p.ev_pct,
                       p.kelly_frac, p.is_value_bet, p.reasoning,
                       g.sport, g.commence_time, g.home_team_id, g.away_team_id
                from predictions p join games g using (event_id)
                where g.commence_time > now() - interval '2 hours'
                  and p.market = 'h2h'
                order by p.event_id, p.market, p.outcome, p.created_at desc
                limit 400
            """))
            bank = pd.DataFrame(query(conn, "select ts, balance from bankroll order by ts"))
        return picks, bank, False
    return _demo_picks(), _demo_bankroll(), True


def _demo_picks() -> pd.DataFrame:
    now = datetime.now()
    rows = [
        dict(event_id="d1", market="h2h", outcome="Argentina", model_prob=0.63,
             ev_pct=-2.1, kelly_frac=0, is_value_bet=False, reasoning=_demo_reason_arg(),
             sport="worldcup", commence_time=now + timedelta(hours=4),
             home_team_id="worldcup_Argentina", away_team_id="worldcup_Egypt"),
        dict(event_id="d1", market="h2h", outcome="Draw", model_prob=0.22,
             ev_pct=-8.0, kelly_frac=0, is_value_bet=False, reasoning=None,
             sport="worldcup", commence_time=now + timedelta(hours=4),
             home_team_id="worldcup_Argentina", away_team_id="worldcup_Egypt"),
        dict(event_id="d1", market="h2h", outcome="Egypt", model_prob=0.15,
             ev_pct=-11.0, kelly_frac=0, is_value_bet=False, reasoning=None,
             sport="worldcup", commence_time=now + timedelta(hours=4),
             home_team_id="worldcup_Argentina", away_team_id="worldcup_Egypt"),
        dict(event_id="d2", market="h2h", outcome="Milwaukee Brewers", model_prob=0.75,
             ev_pct=14.5, kelly_frac=0.05, is_value_bet=True, reasoning=_demo_reason_mil(),
             sport="mlb", commence_time=now + timedelta(hours=7),
             home_team_id="mlb_Milwaukee Brewers", away_team_id="mlb_St. Louis Cardinals"),
        dict(event_id="d2", market="h2h", outcome="St. Louis Cardinals", model_prob=0.25,
             ev_pct=-18.0, kelly_frac=0, is_value_bet=False, reasoning=None,
             sport="mlb", commence_time=now + timedelta(hours=7),
             home_team_id="mlb_Milwaukee Brewers", away_team_id="mlb_St. Louis Cardinals"),
    ]
    return pd.DataFrame(rows)


def _demo_reason_arg() -> str:
    return ("🧠 **MODEL LEAN (NO BET)**\n\n🎯 Argentina to win (90 min) — model 63%.\n\n"
            "💰 **STAKE:** $0 — no betting edge at current prices\n\n⚔️ **TACTICAL EDGE:**\n"
            "- Argentina averaged 2.3 goals per game this tournament vs Egypt's 0.9.\n"
            "- Egypt crossed 2 more time zones and had one fewer rest day.\n\n"
            "⚠️ **RISK CHECK:** Knockout games tighten up; a 63% favorite still loses or "
            "draws 4 times in 10. The market already prices Argentina's dominance.")


def _demo_reason_mil() -> str:
    return ("🎯 **THE PICK:** Milwaukee Brewers ML\n\n📊 **CALCULATED EDGE:** +14.5% EV — "
            "model 75% vs market 66%.\n\n💰 **STAKE:** $50 (5.0% of bankroll, quarter-Kelly)\n\n"
            "⚔️ **TACTICAL EDGE:**\n- Brewers average 5.1 runs/game vs Cardinals' 4.2.\n"
            "- Milwaukee's staff ERA 3.32 vs 4.12.\n\n⚠️ **RISK CHECK:** Baseball is the "
            "highest-variance major sport; even strong edges lose often.")


def _demo_bankroll() -> pd.DataFrame:
    import math
    days = pd.date_range(end=datetime.now(), periods=45)
    bal, curve = 1000.0, []
    for i, d in enumerate(days):
        bal *= 1 + 0.004 * math.sin(i * 1.7) + 0.0022
        curve.append({"ts": d, "balance": round(bal, 2)})
    return pd.DataFrame(curve)


def md_to_html(md: str) -> str:
    """Tiny safe renderer for the analyst briefings (bold + line breaks)."""
    if not md:
        return ""
    # keep sportsbook prices in the backend: strip "@ -205 (betus)" style tokens
    md = re.sub(r"@\s*[+-]?\d+(\.\d+)?\s*\([a-zA-Z0-9_ ]+\)", "", md)
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
    games = []
    if picks.empty:
        return games
    for event_id, grp in picks.groupby("event_id", sort=False):
        g0 = grp.iloc[0]
        home = str(g0["home_team_id"]).split("_")[-1]
        away = str(g0["away_team_id"]).split("_")[-1]
        outs = []
        for _, r in grp.iterrows():
            outs.append({
                "name": r["outcome"],
                "prob": round(float(r["model_prob"] or 0), 4),
                "value": bool(r["is_value_bet"]),
                "stake": round(bankroll * float(r["kelly_frac"] or 0), 2),
                "edge": round(float(r["ev_pct"] or 0), 1),
            })
        order = {home: 0, "Draw": 1, away: 2}
        outs.sort(key=lambda o: order.get(o["name"], 3))
        top = max(outs, key=lambda o: o["prob"])
        pick = next((o for o in outs if o["value"]), None)
        reasoning = next((r for r in grp["reasoning"] if isinstance(r, str) and r.strip()), "")
        games.append({
            "id": str(event_id),
            "league": LEAGUE_LABEL.get(str(g0["sport"]), str(g0["sport"]).upper()),
            "kick": pd.Timestamp(g0["commence_time"]).isoformat(),
            "home": home, "away": away,
            "outcomes": outs,
            "top": top["name"], "topProb": top["prob"],
            "isBet": pick is not None,
            "betName": pick["name"] if pick else None,
            "stake": pick["stake"] if pick else 0,
            "edge": pick["edge"] if pick else None,
            "why": md_to_html(reasoning) or
                   "<i>Analysis being generated — the AI analyst runs hourly.</i>",
        })
    games.sort(key=lambda g: (not g["isBet"], g["kick"]))
    return games


picks, bank, is_demo = load_frames()
balance = float(bank["balance"].iloc[-1]) if not bank.empty else 1000.0
games = build_games(picks, balance)

# ----------------------------------------------------------------- header row

st.markdown(
    f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px;'>"
    f"<span style='font-size:2rem;font-weight:800;letter-spacing:-0.03em;'>📈 VegasEdge</span>"
    f"<span style='color:{ACCENT};font-weight:700;font-size:0.8rem;letter-spacing:0.15em;'>MODEL BOARD</span>"
    f"<span style='color:{INK_MUTED};font-size:0.85rem;'>{datetime.now():%A, %B %d}</span></div>",
    unsafe_allow_html=True)
if is_demo:
    st.caption("⚠️ Demo data — set DATABASE_URL to connect your live database.")

pnl_30 = balance - (float(bank["balance"].iloc[max(0, len(bank) - 30)]) if not bank.empty else balance)
n_bets = sum(1 for g in games if g["isBet"])

c1, c2, c3, c4 = st.columns(4)
for col, label, value, sub, cls in (
    (c1, "Bankroll", f"${balance:,.2f}", f"{pnl_30:+,.2f} last 30 days", "up" if pnl_30 >= 0 else "down"),
    (c2, "Games on the Board", f"{len(games)}", "every game, every outcome", ""),
    (c3, "Model Picks Today", f"{n_bets}", "bets our model backs", "up"),
    (c4, "Staking Discipline", "¼ Kelly", "hard-capped at 5% per bet", ""),
):
    col.markdown(f"<div class='ve-tile'><div class='label'>{label}</div>"
                 f"<div class='value'>{value}</div><div class='sub {cls}'>{sub}</div></div>",
                 unsafe_allow_html=True)

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
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }
  .flip { perspective: 1400px; height: 356px; cursor: pointer; }
  .inner { position: relative; width: 100%; height: 100%;
           transition: transform 0.65s cubic-bezier(0.4, 0.1, 0.2, 1); transform-style: preserve-3d; }
  .flip.flipped .inner { transform: rotateY(180deg); }
  .face { position: absolute; inset: 0; backface-visibility: hidden; border-radius: 18px;
          border: 1px solid #222b38; overflow: hidden;
          background: linear-gradient(165deg, #17202b 0%, #11161e 70%);
          box-shadow: 0 8px 26px rgba(0,0,0,0.35); }
  .flip:hover .face { border-color: #2e3a4a; box-shadow: 0 12px 34px rgba(0,0,0,0.5); }
  .flip.bet .front { border-color: rgba(0,230,138,0.45); }
  .flip.bet:hover .front { box-shadow: 0 12px 34px rgba(0,230,138,0.12); }
  .front { padding: 20px 22px; display: flex; flex-direction: column; }
  .back { transform: rotateY(180deg); padding: 18px 22px; display: flex; flex-direction: column; }
  .rowtop { display: flex; justify-content: space-between; align-items: center; }
  .chip { font-size: 0.66rem; font-weight: 800; letter-spacing: 0.12em; color: #9fb1c4;
          background: rgba(139,152,169,0.12); border: 1px solid #2a3442;
          padding: 4px 10px; border-radius: 999px; }
  .kick { color: #8b98a9; font-size: 0.78rem; font-variant-numeric: tabular-nums; }
  .teams { margin: 14px 0 4px; font-size: 1.22rem; font-weight: 750; letter-spacing: -0.015em;
           color: #e8eef5; line-height: 1.3; }
  .teams .at { color: #55647a; font-weight: 500; font-size: 0.95rem; padding: 0 4px; }
  .verdict { display: flex; align-items: center; gap: 10px; margin: 10px 0 14px; }
  .badge { font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em; padding: 5px 12px;
           border-radius: 999px; }
  .badge.bet { color: #00e68a; background: rgba(0,230,138,0.13); border: 1px solid rgba(0,230,138,0.6);
               animation: pulse 1.8s ease-in-out infinite; }
  .badge.lean { color: #9fb1c4; background: rgba(139,152,169,0.1); border: 1px solid #2a3442; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(0,230,138,0.3); }
                     50% { box-shadow: 0 0 14px 2px rgba(0,230,138,0.22); } }
  .verdict .who { font-size: 1.02rem; font-weight: 700; color: #e8eef5; }
  .verdict .stake { color: #00e68a; font-weight: 700; font-size: 0.9rem; }
  .bars { display: flex; flex-direction: column; gap: 9px; margin-top: auto; }
  .barrow { display: grid; grid-template-columns: 112px 1fr 52px; gap: 10px; align-items: center; }
  .barrow .name { color: #aebbcb; font-size: 0.8rem; white-space: nowrap; overflow: hidden;
                  text-overflow: ellipsis; }
  .barrow .name.win { color: #e8eef5; font-weight: 700; }
  .track { height: 9px; background: #1b2430; border-radius: 999px; overflow: hidden; }
  .fill { height: 100%; border-radius: 999px; transition: width 0.9s cubic-bezier(0.2,0.8,0.2,1); }
  .barrow .pct { color: #e8eef5; font-size: 0.82rem; font-weight: 650; text-align: right;
                 font-variant-numeric: tabular-nums; }
  .tap { margin-top: 14px; color: #55647a; font-size: 0.72rem; letter-spacing: 0.06em;
         text-align: center; }
  .back h4 { color: #00e68a; font-size: 0.72rem; letter-spacing: 0.16em; margin-bottom: 10px; }
  .why { color: #c6d2e0; font-size: 0.82rem; line-height: 1.55; overflow-y: auto; flex: 1;
         padding-right: 6px; }
  .why::-webkit-scrollbar { width: 6px; } .why::-webkit-scrollbar-thumb { background: #2a3442; border-radius: 3px; }
  .why b { color: #e8eef5; }
  .backfoot { margin-top: 10px; color: #55647a; font-size: 0.72rem; text-align: center; }
  .empty { color: #8b98a9; padding: 40px; text-align: center; border: 1px dashed #2a3442;
           border-radius: 16px; }
</style>
<div class="wrap">
  <div class="toolbar">
    <input id="q" class="search" placeholder="🔍  Search any league, team or match…">
    <span class="hint">click a card to see the model's reasoning</span>
  </div>
  <div id="grid" class="grid"></div>
  <div id="empty" class="empty" style="display:none">No games match your search.</div>
</div>
<script>
const GAMES = __GAMES_JSON__;
const COLORS = ["#0284c7", "#d97706", "#9333ea"];  // home, draw, away (fixed order)
const grid = document.getElementById("grid");

function timeStr(iso) {
  const d = new Date(iso);
  const today = new Date(); const isToday = d.toDateString() === today.toDateString();
  const t = d.toLocaleTimeString([], {hour: "numeric", minute: "2-digit"});
  return isToday ? "Today · " + t : d.toLocaleDateString([], {weekday: "short"}) + " · " + t;
}

function card(g) {
  const el = document.createElement("div");
  el.className = "flip" + (g.isBet ? " bet" : "");
  el.dataset.text = (g.league + " " + g.home + " " + g.away).toLowerCase();
  const bars = g.outcomes.map((o, i) => `
    <div class="barrow">
      <div class="name ${o.name === g.top ? "win" : ""}">${o.name === g.top ? "✓ " : ""}${o.name}</div>
      <div class="track"><div class="fill" style="width:0%;background:${COLORS[i % 3]}" data-w="${(o.prob*100).toFixed(1)}"></div></div>
      <div class="pct">${(o.prob*100).toFixed(0)}%</div>
    </div>`).join("");
  const verdict = g.isBet
    ? `<span class="badge bet">🔥 MODEL PICK</span><span class="who">${g.betName}</span>
       <span class="stake">bet $${g.stake.toFixed(0)}</span>`
    : `<span class="badge lean">LEAN · NO BET</span><span class="who">${g.top} ${(g.topProb*100).toFixed(0)}%</span>`;
  el.innerHTML = `
    <div class="inner">
      <div class="face front">
        <div class="rowtop"><span class="chip">${g.league}</span><span class="kick">${timeStr(g.kick)}</span></div>
        <div class="teams">${g.away} <span class="at">@</span> ${g.home}</div>
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

st.markdown("### 🧠 Today's Board — what *our* model says")
if games:
    payload = json.dumps(games).replace("</", "<\\/")
    rows = -(-len(games) // 2)  # assume 2 columns typical width
    height = min(140 + rows * 375, 1600)
    components.html(BOARD_TEMPLATE.replace("__GAMES_JSON__", payload),
                    height=height, scrolling=True)
else:
    st.info("No upcoming games priced right now — the hourly pipeline will fill the board.")

# ----------------------------------------------------------- bankroll section

col_chart, col_next = st.columns([3, 2])
with col_chart:
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
    st.markdown("### 🎯 Today's stakes")
    any_bet = False
    for g in games:
        if g["isBet"]:
            any_bet = True
            st.markdown(
                f"<div class='ve-tile' style='margin-bottom:10px;'>"
                f"<div class='label'>{g['league']} · {g['away']} @ {g['home']}</div>"
                f"<div class='value' style='font-size:1.15rem;'>{g['betName']} — bet ${g['stake']:,.0f}</div>"
                f"<div class='sub up'>model edge +{g['edge']}%</div></div>",
                unsafe_allow_html=True)
    if not any_bet:
        st.markdown(f"<div class='ve-tile'><div class='label'>No bets today</div>"
                    f"<div class='sub'>The model found no mispriced games. Not betting "
                    f"is a position — it's how the bankroll survives.</div></div>",
                    unsafe_allow_html=True)

st.markdown("---")
st.caption("VegasEdge is an analytical tool. Probabilities are model estimates; losses are "
           "expected and normal. Soccer results are for 90 minutes. Bet only where legal and "
           "only what you can afford to lose. 21+.")

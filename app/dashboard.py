"""MODULE 8 — VegasEdge dashboard (Streamlit, free hosting on Community Cloud).

Dark fintech UI: live edge ticker, AI-Brain expandable reasoning cards,
Kelly bankroll tracker, global search. Falls back to demo data when
DATABASE_URL isn't configured so you can preview the UI instantly:

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root importable

st.set_page_config(page_title="VegasEdge", page_icon="📈", layout="wide")

# ---- design tokens (validated for CVD + contrast on #0d1117, dark band) ----
SURFACE = "#0d1117"
PANEL = "#161b22"
BORDER = "#21262d"
INK = "#e6edf3"
INK_MUTED = "#8b949e"
SERIES_GREEN = "#059669"   # chart series
SERIES_CYAN = "#0284c7"
ACCENT = "#00e68a"         # UI accent only (glows/borders), not chart series
ACCENT_CYAN = "#22d3ee"
STATUS_LOSS = "#f85149"

st.markdown(f"""
<style>
  .stApp {{ background: {SURFACE}; }}
  html, body, [class*="css"] {{ color: {INK}; font-family: 'Segoe UI', system-ui, sans-serif; }}
  h1, h2, h3 {{ color: {INK} !important; letter-spacing: -0.02em; }}
  section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}

  .ve-tile {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 20px; height: 100%;
  }}
  .ve-tile .label {{ color: {INK_MUTED}; font-size: 0.72rem; text-transform: uppercase;
                     letter-spacing: 0.08em; margin-bottom: 4px; }}
  .ve-tile .value {{ color: {INK}; font-size: 1.7rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
  .ve-tile .delta-up {{ color: {ACCENT}; font-size: 0.85rem; }}
  .ve-tile .delta-down {{ color: {STATUS_LOSS}; font-size: 0.85rem; }}

  .ve-ticker-row {{
    display: grid; grid-template-columns: 2.2fr 1fr 1fr 1fr 0.9fr; gap: 8px;
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px; align-items: center;
    font-variant-numeric: tabular-nums;
  }}
  .ve-ticker-head {{ background: transparent; border: none; color: {INK_MUTED};
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 4px 16px; }}
  .ve-match {{ font-weight: 600; }} .ve-sub {{ color: {INK_MUTED}; font-size: 0.78rem; }}
  .ve-num {{ color: {INK}; }} .ve-edge-pos {{ color: {ACCENT}; font-weight: 700; }}
  .ve-badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
    font-weight: 700; text-align: center;
  }}
  .ve-badge.hot {{
    background: rgba(0,230,138,0.12); color: {ACCENT}; border: 1px solid {ACCENT};
    animation: ve-pulse 1.6s ease-in-out infinite;
  }}
  .ve-badge.warm {{ background: rgba(34,211,238,0.10); color: {ACCENT_CYAN};
                    border: 1px solid rgba(34,211,238,0.4); }}
  .ve-badge.pass {{ background: rgba(139,148,158,0.1); color: {INK_MUTED};
                    border: 1px solid {BORDER}; }}
  @keyframes ve-pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(0,230,138,0.35); }}
    50% {{ box-shadow: 0 0 12px 2px rgba(0,230,138,0.25); }}
  }}
  div[data-testid="stExpander"] {{ background: {PANEL}; border: 1px solid {BORDER};
    border-radius: 10px; }}
  .stTextInput input {{ background: {PANEL}; color: {INK}; border: 1px solid {BORDER};
    border-radius: 10px; }}
  .stTextInput input:focus {{ border-color: {ACCENT_CYAN}; box-shadow: 0 0 0 1px {ACCENT_CYAN}; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data layer

@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """(picks, bankroll_curve, is_demo)."""
    try:
        secret_url = st.secrets.get("DATABASE_URL", "")
    except Exception:   # no secrets.toml in local dev
        secret_url = ""
    if os.environ.get("DATABASE_URL") or secret_url:
        os.environ.setdefault("DATABASE_URL", secret_url)
        from src.utils.db import get_conn, query
        with get_conn() as conn:
            picks = pd.DataFrame(query(conn, """
                select p.*, g.sport, g.commence_time, g.home_team_id, g.away_team_id
                from predictions p join games g using (event_id)
                where g.commence_time > now() - interval '6 hours'
                order by p.ev_pct desc nulls last limit 200
            """))
            bank = pd.DataFrame(query(conn, "select ts, balance from bankroll order by ts"))
        return picks, bank, False
    # -------- demo data so the UI runs with zero setup --------
    now = datetime.now()
    picks = pd.DataFrame([
        dict(pred_id=1, sport="nba", market="h2h", outcome="Boston Celtics", line=None,
             home_team_id="nba_Boston Celtics", away_team_id="nba_Miami Heat",
             commence_time=now + timedelta(hours=5), model_prob=0.62, fair_decimal=1.61,
             best_book="draftkings", best_decimal=1.78, ev_pct=10.4, kelly_frac=0.031,
             is_value_bet=True, reasoning=_demo_reasoning()),
        dict(pred_id=2, sport="epl", market="h2h", outcome="Arsenal", line=None,
             home_team_id="epl_Arsenal", away_team_id="epl_Chelsea",
             commence_time=now + timedelta(hours=26), model_prob=0.51, fair_decimal=1.96,
             best_book="fanduel", best_decimal=2.10, ev_pct=7.1, kelly_frac=0.017,
             is_value_bet=True, reasoning="🎯 **THE PICK:** Arsenal ML @ +110 (fanduel)…"),
        dict(pred_id=3, sport="nfl", market="h2h", outcome="Buffalo Bills", line=None,
             home_team_id="nfl_Buffalo Bills", away_team_id="nfl_New York Jets",
             commence_time=now + timedelta(hours=49), model_prob=0.68, fair_decimal=1.47,
             best_book="betmgm", best_decimal=1.50, ev_pct=2.0, kelly_frac=0.005,
             is_value_bet=False, reasoning="🚫 **VERDICT: PASS** — market already prices Bills at our number."),
    ])
    days = pd.date_range(end=now, periods=45)
    bal, curve = 1000.0, []
    import math
    for i, d in enumerate(days):
        bal *= 1 + 0.004 * math.sin(i * 1.7) + 0.0022
        curve.append({"ts": d, "balance": round(bal, 2)})
    return picks, pd.DataFrame(curve), True


def _demo_reasoning() -> str:
    return """🔥 **HIGH-VALUE ALERT**
🎯 **THE PICK:** Boston Celtics Moneyline @ -128 (draftkings)

📊 **CALCULATED EDGE:** +10.4% EV — model gives Boston a 62% chance vs. the 56% the market is pricing in.

💰 **STAKE:** $31.00 (3.1% of bankroll, quarter-Kelly)

⚔️ **TACTICAL EDGE:**
- Boston's net rating edge is +6.8 points per 100 possessions over Miami.
- Miami is on a back-to-back with 2,100 km of travel; Boston has had 3 days of rest.
- Sharp-money signal: the line moved toward Boston across five books within one hour (steam move).

⚠️ **RISK CHECK:** Edges this large are rare and often mean the model is missing news the market has — verify injuries before betting. A 62% favorite still loses 4 times in 10; this is a probabilistic edge, not a certainty."""


picks, bank, is_demo = load_data()

# ------------------------------------------------------------------ header

left, right = st.columns([3, 2])
with left:
    st.markdown(f"## 📈 VegasEdge <span style='color:{ACCENT};font-size:0.9rem;'>LIVE</span>",
                unsafe_allow_html=True)
    if is_demo:
        st.caption("⚠️ Demo data — set DATABASE_URL to connect your Supabase instance.")
with right:
    search = st.text_input("Search", placeholder="🔍  Search any league, match, team or player…",
                           label_visibility="collapsed")

if search and not picks.empty:
    s = search.lower()
    picks = picks[picks.apply(
        lambda r: s in str(r["outcome"]).lower() or s in str(r["sport"]).lower()
        or s in str(r["home_team_id"]).lower() or s in str(r["away_team_id"]).lower(), axis=1)]

# ------------------------------------------------------------------ stat tiles

balance = bank["balance"].iloc[-1] if not bank.empty else 1000.0
pnl_30d = balance - (bank["balance"].iloc[max(0, len(bank) - 30)] if not bank.empty else balance)
n_value = int(picks["is_value_bet"].sum()) if not picks.empty else 0
vb_only = picks[picks["is_value_bet"]] if not picks.empty else picks
best_ev = vb_only["ev_pct"].max() if not vb_only.empty else 0.0
if not picks.empty:  # value bets first (best edge on top); PASS rows sorted by
    # distance from market so model-error outliers sink to the bottom
    picks = pd.concat([
        picks[picks["is_value_bet"]].sort_values("ev_pct", ascending=False),
        picks[~picks["is_value_bet"]].sort_values("ev_pct", key=lambda s: s.abs()),
    ])

def tile(col, label, value, delta=None, up=True):
    d = (f"<div class='delta-{'up' if up else 'down'}'>{delta}</div>" if delta else "")
    col.markdown(f"<div class='ve-tile'><div class='label'>{label}</div>"
                 f"<div class='value'>{value}</div>{d}</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
tile(c1, "Bankroll", f"${balance:,.2f}", f"{pnl_30d:+,.2f} last 30d", pnl_30d >= 0)
tile(c2, "Live Value Bets", n_value, "EV > 2% after devig")
tile(c3, "Best Edge Now", f"+{best_ev:.1f}%", "vs best book price")
tile(c4, "Staking Mode", "¼ Kelly", "capped at 5% / bet")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------ edge ticker

st.markdown("### ⚡ Vegas Edge — Live Ticker")
st.markdown(
    "<div class='ve-ticker-row ve-ticker-head'><div>Matchup / Pick</div>"
    "<div>Book Line</div><div>AI True Line</div><div>Edge (EV)</div><div>Signal</div></div>",
    unsafe_allow_html=True)

def american(dec: float) -> str:
    a = round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))
    return f"+{a}" if a > 0 else str(a)

if picks.empty:
    st.info("No priced events right now — the hourly pipeline will populate this feed.")
else:
    for _, r in picks.head(25).iterrows():
        away = str(r["away_team_id"]).split("_", 1)[-1]
        home = str(r["home_team_id"]).split("_", 1)[-1]
        ev = r["ev_pct"] or 0
        badge = ("<span class='ve-badge hot'>🔥 VALUE</span>" if r["is_value_bet"] and ev >= 6
                 else "<span class='ve-badge warm'>VALUE</span>" if r["is_value_bet"]
                 else "<span class='ve-badge pass'>PASS</span>")
        edge_cls = "ve-edge-pos" if r["is_value_bet"] else "ve-sub"
        st.markdown(f"""
        <div class='ve-ticker-row'>
          <div><span class='ve-match'>{away} @ {home}</span><br>
               <span class='ve-sub'>{str(r['sport']).upper()} · pick: {r['outcome']} ({r['market']})</span></div>
          <div class='ve-num'>{american(r['best_decimal'])}<br><span class='ve-sub'>{r['best_book']}</span></div>
          <div class='ve-num'>{american(r['fair_decimal'])}<br><span class='ve-sub'>p = {r['model_prob']:.0%}</span></div>
          <div class='{edge_cls}'>{'+' if ev >= 0 else ''}{ev:.1f}%</div>
          <div>{badge}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------ AI brain + bankroll

col_brain, col_bank = st.columns([3, 2])

with col_brain:
    st.markdown("### 🧠 AI Brain — Analyst Briefings")
    vb = picks[picks["is_value_bet"]] if not picks.empty else picks
    if vb.empty:
        st.caption("No value bets flagged right now.")
    for _, r in vb.head(8).iterrows():
        stake = balance * (r["kelly_frac"] or 0)
        title = (f"🎯 {r['outcome']} ({str(r['sport']).upper()}) · "
                 f"+{(r['ev_pct'] or 0):.1f}% EV · stake ${stake:,.0f}")
        with st.expander(title):
            st.markdown(r["reasoning"] or "_Analysis pending — the LLM agent runs hourly._")

with col_bank:
    st.markdown("### 💰 Bankroll Tracker")
    if not bank.empty:
        fig = go.Figure(go.Scatter(
            x=bank["ts"], y=bank["balance"], mode="lines",
            line=dict(color=SERIES_GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(5,150,105,0.10)",
            hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
        ))
        fig.update_layout(
            template=None, height=260, margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color=INK_MUTED, gridcolor=BORDER, showgrid=False),
            yaxis=dict(color=INK_MUTED, gridcolor=BORDER, tickprefix="$",
                       rangemode="tozero" if bank["balance"].min() < 200 else "normal"),
            hoverlabel=dict(bgcolor=PANEL, font_color=INK),
        )
        fig.update_yaxes(autorange=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("#### Next wagers (quarter-Kelly)")
    if not vb.empty:
        for _, r in vb.head(5).iterrows():
            stake = balance * (r["kelly_frac"] or 0)
            st.markdown(
                f"<div class='ve-tile' style='margin-bottom:8px;'>"
                f"<div class='label'>{r['outcome']} · {american(r['best_decimal'])}</div>"
                f"<div class='value' style='font-size:1.2rem;'>Bet ${stake:,.2f}"
                f"<span class='ve-sub'>  ({(r['kelly_frac'] or 0):.1%} of bankroll)</span></div></div>",
                unsafe_allow_html=True)
    st.caption("Sizing = ¼ Kelly on model probability vs best available price, "
               "hard-capped at 5% of bankroll. Never bet more.")

st.markdown("---")
st.caption("VegasEdge is an analytical tool. Probabilities are estimates; losses are expected "
           "and normal. Bet only where legal and only what you can afford to lose. 21+.")

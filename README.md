# VegasEdge — Free-Tier Sports Betting Prediction Engine

An end-to-end, $0/month sports prediction system: automated data ingestion, an XGBoost
probability engine, Poisson goal models, EV + Kelly bet selection, market-anomaly
detection, an LLM reasoning layer, a self-correcting feedback loop, and a dark-mode
Streamlit trading dashboard.

> **Reality check (read this first):** Sportsbook closing lines are ~96–97% efficient.
> No model prints money on main NFL/NBA spreads. The profitable strategy this repo
> implements is *selective*: find the small fraction of mispriced lines (player props,
> niche markets, early openers, stale lines during news), bet only when EV > 0 after
> removing the bookmaker's vig, and size with fractional Kelly. Expect long variance.
> Bet legally and only what you can afford to lose. This is an analytical tool, not
> financial advice.

---

## MODULE 1 — Free Tech Stack & Architecture

```
┌────────────────────────── GitHub Actions (free, cron) ──────────────────────────┐
│  hourly:  ingest_odds  ─►  ingest_stats  ─►  deep_data (weather/sentiment)      │
│  hourly:  predict ─► value_engine ─► llm_analyst ─► write picks to DB           │
│  daily :  settle_and_learn (grade bets, P/L, recalibrate model)                  │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │  Supabase Postgres (free 500MB)│
                     │  time-series odds snapshots,   │
                     │  games, predictions, bankroll  │
                     └───────────────┬───────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │ Streamlit Community Cloud (free)│
                     │ dark fintech dashboard          │
                     └───────────────────────────────┘
```

### Free-tier bill of materials

| Layer            | Service / Library                         | Free limit                     |
|------------------|-------------------------------------------|--------------------------------|
| Compute / cron   | **GitHub Actions**                        | 2,000 min/mo (public repo: unlimited) |
| Database         | **Supabase Postgres** (or Neon)           | 500 MB, 2 projects             |
| Odds (live lines)| **The Odds API** (the-odds-api.com)       | 500 credits/mo — budget them   |
| Schedules/scores | **ESPN hidden JSON API** (no key)         | unlimited (be polite)          |
| NBA stats        | **`nba_api`** (official stats.nba.com)    | free                           |
| NFL stats (EPA)  | **`nfl_data_py`** (nflverse play-by-play) | free                           |
| MLB stats        | **`pybaseball`** (FanGraphs/Statcast)     | free                           |
| Soccer xG        | **Understat scrape** / football-data.org  | free / 10 req-min free key     |
| NHL stats        | **NHL Stats API** (api-web.nhle.com)      | free, no key                   |
| Injuries         | ESPN injuries JSON + Rotowire RSS         | free                           |
| Weather          | **Open-Meteo**                            | free, no key                   |
| Sentiment        | Reddit public JSON + RSS + VADER          | free                           |
| LLM reasoning    | **Groq API** (Llama 3.3 70B) or **Gemini**| generous free tiers            |
| Scraping         | `httpx`, `BeautifulSoup4`, `playwright`   | open source                    |
| ML               | `xgboost`, `scikit-learn`, `pandas`       | open source                    |
| Frontend         | **Streamlit Community Cloud**             | free hosting                   |

### Database design notes (time-series optimized)
- `odds_snapshots` is append-only with a `captured_at` timestamp — this is your
  time-series of line movement. Index `(event_id, market, captured_at DESC)`.
- Opening line vs. current line = sharp-money signal (Module 4).
- Keep raw JSON payloads in `JSONB` columns so you never lose data to schema changes.
- 500 MB goes far: an odds snapshot row is ~200 bytes; hourly snapshots of 100 events
  across 4 markets ≈ 35 MB/year. Prune snapshots > 18 months with the included job.

### Repo layout
```
db/schema.sql                 Postgres schema (run once in Supabase SQL editor)
.github/workflows/            free cron pipelines (hourly ingest, daily settle)
src/ingestion/                Module 1 & 7: odds, stats, injuries, weather, sentiment
src/features/                 Module 2: preprocessing & advanced metrics
src/models/                   Module 2 & 3: XGBoost engine + Poisson models
src/betting/                  Module 3 & 4: EV, Kelly, market anomalies, tournaments
src/llm/                      Module 5: AI Sports Analyst prompt + agent
src/feedback/                 Module 6: settle results, log P/L, auto-recalibrate
src/utils/                    DB pool + odds math primitives
app/dashboard.py              Module 8: Streamlit fintech dashboard
```

---

## Step-by-step setup (all free)

1. **Database** — create a free project at [supabase.com](https://supabase.com).
   Open the SQL editor, paste `db/schema.sql`, run it. Copy the connection string
   (Settings → Database → URI, use the *pooler* URI for GitHub Actions).
2. **API keys** — sign up free at [the-odds-api.com](https://the-odds-api.com) (500
   credits/mo) and [console.groq.com](https://console.groq.com) (free LLM inference).
3. **Local dev** —
   ```bash
   python -m venv .venv && .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env    # fill in DATABASE_URL, ODDS_API_KEY, GROQ_API_KEY
   python -m src.ingestion.odds_ingest        # pull lines
   python -m src.ingestion.stats_ingest nba   # pull team stats
   python -m src.models.train nba             # train baseline model
   python -m src.betting.value_engine         # generate value bets
   streamlit run app/dashboard.py             # open the dashboard
   ```
4. **Automation** — push to GitHub, then add repo secrets
   (`Settings → Secrets → Actions`): `DATABASE_URL`, `ODDS_API_KEY`, `GROQ_API_KEY`.
   The workflows in `.github/workflows/` start running on schedule automatically.
5. **Dashboard hosting** — share.streamlit.io → deploy `app/dashboard.py` from your
   repo, add the same secrets in the Streamlit secrets panel.

### Budgeting the 500 free Odds API credits
One request per sport per market region costs 1 credit × markets. Strategy:
- Poll **h2h+spreads+totals** for your 2 focus sports every 2 hours on game days only
  (the workflow checks the schedule first) ≈ 360 credits/mo.
- Player props cost more credits — pull them once daily for games you already
  flagged as interesting.

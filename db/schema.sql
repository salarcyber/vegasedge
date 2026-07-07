-- VegasEdge schema — run once in the Supabase SQL editor.
-- Optimized for append-only time-series odds data on a 500 MB free tier.

create table if not exists teams (
    team_id      text primary key,          -- e.g. 'nba_BOS', 'epl_ARS'
    sport        text not null,             -- nba | nfl | mlb | nhl | soccer_epl | soccer_wc
    name         text not null,
    abbrev       text,
    venue_lat    double precision,          -- for travel-distance features (Module 7)
    venue_lon    double precision,
    venue_tz     text                       -- IANA tz, for circadian/travel features
);

create table if not exists players (
    player_id    text primary key,
    team_id      text references teams(team_id),
    sport        text not null,
    name         text not null,
    position     text
);

create table if not exists games (
    event_id     text primary key,          -- The Odds API event id (or espn id prefixed)
    sport        text not null,
    league       text,
    commence_time timestamptz not null,
    home_team_id text references teams(team_id),
    away_team_id text references teams(team_id),
    status       text default 'scheduled',  -- scheduled | live | final
    home_score   int,
    away_score   int,
    -- tournament context (Module 4)
    stage        text,                      -- group | knockout | final ...
    home_rest_days int,
    away_rest_days int,
    home_travel_km double precision,
    away_travel_km double precision,
    raw          jsonb
);
create index if not exists idx_games_sport_time on games (sport, commence_time desc);

-- Append-only line history: THE core time-series table.
create table if not exists odds_snapshots (
    id           bigint generated always as identity primary key,
    event_id     text not null references games(event_id),
    bookmaker    text not null,
    market       text not null,             -- h2h | spreads | totals | player_points ...
    outcome      text not null,             -- team name / Over / Under / player name
    line         double precision,          -- spread or total number, null for h2h
    price_american int not null,
    price_decimal double precision not null,
    captured_at  timestamptz not null default now(),
    is_opening   boolean default false
);
create index if not exists idx_snap_lookup on odds_snapshots (event_id, market, captured_at desc);

-- Public betting % vs money % when scrapeable (Module 4 anomaly detection).
create table if not exists public_betting (
    event_id     text references games(event_id),
    market       text,
    side         text,
    bet_pct      double precision,          -- % of tickets
    money_pct    double precision,          -- % of handle
    captured_at  timestamptz default now(),
    primary key (event_id, market, side, captured_at)
);

-- Team-level rolling advanced metrics (Module 2 features).
create table if not exists team_metrics (
    team_id      text references teams(team_id),
    as_of        date not null,
    metrics      jsonb not null,            -- {net_rating, off_rtg, pace, epa_off, xg_for, wrc_plus,...}
    primary key (team_id, as_of)
);

create table if not exists player_metrics (
    player_id    text references players(player_id),
    as_of        date not null,
    metrics      jsonb not null,            -- rolling per-game splits, usage, minutes proj
    primary key (player_id, as_of)
);

create table if not exists injuries (
    player_id    text references players(player_id),
    reported_at  timestamptz default now(),
    status       text,                      -- out | doubtful | questionable | probable
    detail       text,
    source       text,
    primary key (player_id, reported_at)
);

-- Module 7: hyper-granular context
create table if not exists game_context (
    event_id     text primary key references games(event_id),
    weather      jsonb,                     -- {temp_c, wind_kph, wind_dir_deg, humidity, pressure_hpa, precip_mm}
    referee      text,
    sentiment    jsonb,                     -- {home_score:-1..1, away_score:-1..1, n_docs, top_topics}
    tz_crossed_home int default 0,
    tz_crossed_away int default 0,
    notes        jsonb
);

-- Situational splits: performance under a specific referee / rest state / surface etc.
create table if not exists situational_splits (
    subject_id   text not null,             -- team_id or player_id
    split_key    text not null,             -- 'referee:Scott Foster' | 'rest:b2b' | 'surface:turf'
    n            int not null,
    value        jsonb not null,            -- {win_pct, ppg_delta, cover_pct, ...}
    as_of        date not null,
    primary key (subject_id, split_key, as_of)
);

-- Model output + bet ledger (Modules 3, 5, 6)
create table if not exists predictions (
    pred_id      bigint generated always as identity primary key,
    event_id     text references games(event_id),
    market       text not null,
    outcome      text not null,
    line         double precision,
    model_prob   double precision not null, -- calibrated "true" probability
    fair_decimal double precision not null, -- 1/model_prob
    best_book    text,
    best_decimal double precision,
    ev_pct       double precision,          -- (p*dec - 1) * 100
    kelly_frac   double precision,          -- fraction of bankroll (already fractional-kelly'd)
    model_version text,
    reasoning    text,                      -- Module 5 LLM plain-English analysis
    created_at   timestamptz default now(),
    is_value_bet boolean default false
);
create index if not exists idx_pred_event on predictions (event_id, created_at desc);

create table if not exists bet_results (
    pred_id      bigint primary key references predictions(pred_id),
    stake        double precision,
    result       text,                      -- win | loss | push | void
    pnl          double precision,
    closing_decimal double precision,       -- for CLV (closing line value) tracking
    clv_pct      double precision,
    settled_at   timestamptz default now()
);

create table if not exists bankroll (
    ts           timestamptz primary key default now(),
    balance      double precision not null
);

-- Model performance ledger for auto-recalibration (Module 6)
create table if not exists model_calibration (
    model_version text,
    sport        text,
    window_end   date,
    n_bets       int,
    hit_rate     double precision,
    roi          double precision,
    brier        double precision,
    logloss      double precision,
    prob_shrink  double precision,          -- learned shrinkage toward market (0..1)
    primary key (model_version, sport, window_end)
);

-- housekeeping: prune old snapshots to stay under free tier
create or replace function prune_old_snapshots() returns void language sql as $$
    delete from odds_snapshots where captured_at < now() - interval '18 months';
$$;

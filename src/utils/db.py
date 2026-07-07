"""Shared Postgres helpers (Supabase/Neon free tier).

Uses a plain psycopg connection per script run — GitHub Actions jobs are
short-lived, so a pool buys nothing and the Supabase pooler handles reuse.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv

load_dotenv()


@contextmanager
def get_conn():
    url = os.environ["DATABASE_URL"]
    try:
        conn = psycopg.connect(url, autocommit=False, connect_timeout=15)
    except psycopg.OperationalError:
        # Some networks/VPNs block 5432; Supabase's transaction pooler on 6543
        # usually gets through. It doesn't support prepared statements, so
        # disable them for the fallback connection.
        fallback = url.replace(":5432/", ":6543/")
        conn = psycopg.connect(fallback, autocommit=False, connect_timeout=15,
                               prepare_threshold=None)
    try:
        with conn:
            yield conn
            conn.commit()
    finally:
        conn.close()


def upsert(conn, table: str, rows: list[dict], conflict_cols: list[str]) -> int:
    """Generic batch upsert. JSON-serializes dict/list values for jsonb columns."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in conflict_cols)
    conflict = ", ".join(conflict_cols)
    action = f"do update set {updates}" if updates else "do nothing"
    sql = (
        f"insert into {table} ({', '.join(cols)}) values ({placeholders}) "
        f"on conflict ({conflict}) {action}"
    )
    vals = [
        tuple(json.dumps(r[c]) if isinstance(r[c], (dict, list)) else r[c] for c in cols)
        for r in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, vals)
    return len(rows)


def insert_many(conn, table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"insert into {table} ({', '.join(cols)}) values ({placeholders})"
    vals = [
        tuple(json.dumps(r[c]) if isinstance(r[c], (dict, list)) else r[c] for c in cols)
        for r in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, vals)
    return len(rows)


def query(conn, sql: str, params: tuple = ()) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        colnames = [d.name for d in cur.description]
        return [dict(zip(colnames, row)) for row in cur.fetchall()]

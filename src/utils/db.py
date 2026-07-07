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
    with psycopg.connect(url, autocommit=False) as conn:
        yield conn
        conn.commit()


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

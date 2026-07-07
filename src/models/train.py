"""CLI: python -m src.models.train nba  — trains the sport's model on all
settled games in the DB and saves the artifact + calibration report."""
from __future__ import annotations

import sys

from src.features.preprocess import build_matchup_frame
from src.models.predictor import train
from src.utils.db import get_conn, upsert


def main(sport: str) -> None:
    with get_conn() as conn:
        df = build_matchup_frame(conn, sport, include_labels=True)
        if len(df) < 100:
            print(f"[train] only {len(df)} settled {sport} games — need 100+. "
                  "Backfill history first (run ingestion for past seasons).")
            return
        report = train(df, sport)
        upsert(conn, "model_calibration", [{
            "model_version": report.model_version, "sport": sport,
            "window_end": __import__("datetime").date.today(),
            "n_bets": 0, "hit_rate": None, "roi": None,
            "brier": report.brier, "logloss": report.logloss,
            "prob_shrink": 0.5,
        }], ["model_version", "sport", "window_end"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nba")

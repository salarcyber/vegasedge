"""MODULE 2 — Core prediction engine: calibrated XGBoost.

Design decisions that matter more than the algorithm choice:

  1. CALIBRATION IS EVERYTHING. A model that's 60% accurate but says 75% loses
     money on every bet. We wrap XGBoost in isotonic CalibratedClassifierCV so
     predicted probabilities match observed frequencies — EV math (Module 3) is
     only valid on calibrated probabilities.
  2. TIME-SERIES SPLITS ONLY. Random CV leaks future info (season trends,
     roster moves). TimeSeriesSplit respects chronology.
  3. SHRINK TOWARD THE MARKET. Final prob = w * model + (1 - w) * market prior.
     w ("prob_shrink") is learned by Module 6 from rolling results — when the
     model runs cold, it automatically defers to the market more.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocess import feature_columns, make_preprocessor

MODEL_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class TrainReport:
    sport: str
    n_games: int
    brier: float
    logloss: float
    market_brier: float | None   # baseline: the opening line's own Brier score
    model_version: str


def build_model(sport: str) -> Pipeline:
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,              # shallow trees: sports data is noisy, deep trees memorize
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        eval_metric="logloss",
        n_jobs=-1,
    )
    return Pipeline([
        ("prep", make_preprocessor(sport)),
        ("clf", CalibratedClassifierCV(xgb, method="isotonic", cv=3)),
    ])


def train(df: pd.DataFrame, sport: str, target: str = "home_win") -> TrainReport:
    df = df.dropna(subset=[target]).sort_values("event_id").reset_index(drop=True)
    X, y = df[feature_columns(sport)], df[target].astype(int)

    # walk-forward validation for an honest generalization estimate
    tss = TimeSeriesSplit(n_splits=4)
    briers, lls = [], []
    for tr_idx, te_idx in tss.split(X):
        m = build_model(sport)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = m.predict_proba(X.iloc[te_idx])[:, 1]
        briers.append(brier_score_loss(y.iloc[te_idx], p))
        lls.append(log_loss(y.iloc[te_idx], p, labels=[0, 1]))

    # the bar to clear: is the model better-calibrated than the opening line?
    market_brier = None
    mkt = df["market_home_prob_open"].dropna()
    if len(mkt) > 50:
        market_brier = float(brier_score_loss(y.loc[mkt.index], mkt))

    final = build_model(sport)
    final.fit(X, y)
    version = f"{sport}-{target}-v{pd.Timestamp.utcnow():%Y%m%d}"
    joblib.dump(final, MODEL_DIR / f"{version}.joblib")
    (MODEL_DIR / f"{sport}-{target}-latest.txt").write_text(version)

    report = TrainReport(
        sport=sport, n_games=len(df),
        brier=round(float(np.mean(briers)), 4),
        logloss=round(float(np.mean(lls)), 4),
        market_brier=round(market_brier, 4) if market_brier else None,
        model_version=version,
    )
    print(f"[train] {json.dumps(report.__dict__)}")
    if report.market_brier and report.brier > report.market_brier:
        print("[train] WARNING: model is WORSE calibrated than the market opener. "
              "Do not bet h2h with this model; rely on shrinkage / niche markets.")
    return report


def load_latest(sport: str, target: str = "home_win") -> Pipeline:
    version = (MODEL_DIR / f"{sport}-{target}-latest.txt").read_text().strip()
    return joblib.load(MODEL_DIR / f"{version}.joblib"), version


def predict_true_prob(
    df_upcoming: pd.DataFrame, sport: str, prob_shrink: float = 0.5
) -> pd.DataFrame:
    """Calibrated home-win probability, shrunk toward the market prior.

    prob_shrink comes from model_calibration (Module 6): weight on OUR model.
    0.5 is a humble default; it grows only when live results prove the model.
    """
    model, version = load_latest(sport)
    raw = model.predict_proba(df_upcoming[feature_columns(sport)])[:, 1]
    market = df_upcoming["market_home_prob_open"].fillna(pd.Series(raw)).to_numpy(dtype=float)
    blended = prob_shrink * raw + (1 - prob_shrink) * market
    out = df_upcoming[["event_id"]].copy()
    out["model_prob_raw"] = np.round(raw, 4)
    out["model_prob"] = np.round(blended, 4)
    out["model_version"] = version
    return out

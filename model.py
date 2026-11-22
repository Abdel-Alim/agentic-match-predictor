"""Model training, honest evaluation, persistence and prediction.

Evaluation uses a time-based holdout: train on completed seasons, evaluate
on the current season's completed matches. Metrics reported on the site are
real numbers computed on data the model never trained on, alongside two
baselines for context (majority class and bookmaker-implied probabilities).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import HOLDOUT_SEASON, MODEL_DIR
from features import FEATURE_COLUMNS

CLASSES = ["H", "D", "A"]  # pipeline classes_ order is alphabetical: ['A','D','H']
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metrics: dict
    trained_at: str


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba)
    for r, y in enumerate(y_true):
        onehot[r, idx[y]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _bookmaker_baseline(holdout: pd.DataFrame, classes: list[str]) -> dict | None:
    sub = holdout.dropna(subset=["odd_home", "odd_draw", "odd_away"])
    if len(sub) < 30:
        return None
    raw = sub[["odd_home", "odd_draw", "odd_away"]].to_numpy(dtype=float)
    raw = np.where(raw <= 1.0, np.nan, raw)
    implied = 1.0 / raw
    implied = implied / implied.sum(axis=1, keepdims=True)  # remove overround
    order = {"H": 0, "D": 1, "A": 2}
    proba = implied[:, [order[c] for c in classes]]
    y = sub["result"].to_numpy()
    return {
        "n": int(len(sub)),
        "log_loss": round(float(log_loss(y, proba, labels=classes)), 4),
        "accuracy": round(float(accuracy_score(y, sub[["odd_home", "odd_draw", "odd_away"]].idxmin(axis=1).map(
            {"odd_home": "H", "odd_draw": "D", "odd_away": "A"}))), 4),
    }


def train(features: pd.DataFrame) -> ModelBundle:
    df = features.dropna(subset=FEATURE_COLUMNS + ["result"]).copy()
    train_df = df[df["season"] != HOLDOUT_SEASON]
    holdout_df = df[(df["season"] == HOLDOUT_SEASON)]

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )
    pipe.fit(train_df[FEATURE_COLUMNS], train_df["result"])
    classes = list(pipe.classes_)  # ['A', 'D', 'H']

    metrics: dict = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_matches": int(len(train_df)),
        "holdout_matches": int(len(holdout_df)),
        "holdout_season": HOLDOUT_SEASON,
        "classes": classes,
        "features": FEATURE_COLUMNS,
    }
    if len(holdout_df) >= 30:
        proba = pipe.predict_proba(holdout_df[FEATURE_COLUMNS])
        y = holdout_df["result"].to_numpy()
        majority = train_df["result"].value_counts(normalize=True)
        maj_proba = np.tile(
            [majority.get(c, 0.0) for c in classes], (len(holdout_df), 1)
        )
        metrics["holdout"] = {
            "accuracy": round(float(accuracy_score(y, pipe.predict(holdout_df[FEATURE_COLUMNS]))), 4),
            "log_loss": round(float(log_loss(y, proba, labels=classes)), 4),
            "brier": round(_multiclass_brier(y, proba, classes), 4),
        }
        metrics["baseline_majority"] = {
            "log_loss": round(float(log_loss(y, maj_proba, labels=classes)), 4),
            "accuracy": round(float(accuracy_score(y, [train_df["result"].mode()[0]] * len(y))), 4),
        }
        bm = _bookmaker_baseline(holdout_df, classes)
        if bm:
            metrics["baseline_bookmaker"] = bm

    trained_at = metrics["trained_at"]
    return ModelBundle(pipe, metrics, trained_at)


def save(bundle: ModelBundle) -> None:
    joblib.dump(bundle.pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(bundle.metrics, indent=2))


def load() -> ModelBundle | None:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return None
    metrics = json.loads(METRICS_PATH.read_text())
    return ModelBundle(joblib.load(MODEL_PATH), metrics, metrics.get("trained_at", ""))


def model_age_hours(bundle: ModelBundle) -> float:
    try:
        trained = dt.datetime.fromisoformat(bundle.trained_at)
    except ValueError:
        return float("inf")
    return (dt.datetime.now(dt.timezone.utc) - trained).total_seconds() / 3600.0


def predict_proba(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    """Rows aligned to CLASSES order -> (n, 3) array in [H, D, A] order."""
    classes = list(bundle.pipeline.classes_)
    proba = bundle.pipeline.predict_proba(X[FEATURE_COLUMNS])
    order = [classes.index(c) for c in CLASSES]
    return proba[:, order]

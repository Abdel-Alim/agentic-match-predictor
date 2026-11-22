"""Pipeline tests: run with `python -m pytest` from the repo root."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model as model_mod
from agents import critic_agent, predictor_agent
from features import FEATURE_COLUMNS, LeagueState, build_training_frame
from teams import TeamResolver, normalize


def _toy_history(n_per_pair: int = 6) -> pd.DataFrame:
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    rows = []
    base = dt.date(2025, 8, 1)
    i = 0
    for round_ in range(n_per_pair * 2):
        season = "2425" if round_ < n_per_pair else "2526"
        for h, a in [("Alpha", "Beta"), ("Gamma", "Delta"), ("Beta", "Gamma"), ("Delta", "Alpha")]:
            hg, ag = (0, 2) if i % 5 == 0 else ((2, 1) if i % 3 else (1, 1))
            rows.append({
                "league": "T0", "league_name": "Toy League", "season": season,
                "date": pd.Timestamp(base + dt.timedelta(days=round_ * 3)),
                "home": h, "away": a, "home_goals": hg, "away_goals": ag,
                "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                "odd_home": 2.0, "odd_draw": 3.4, "odd_away": 3.8,
            })
            i += 1
    return pd.DataFrame(rows)


def test_elo_rewards_winner_and_never_leaks_future():
    state = LeagueState()
    before = state.features_before("Alpha", "Beta", "2526")
    state.update("Alpha", "Beta", 3, 0, "2526")
    after = state.features_before("Alpha", "Beta", "2526")
    assert after["elo_diff"] > before["elo_diff"]
    # snapshot taken before update must be unchanged by the result
    assert before["home_matches"] == 0


def test_training_frame_has_no_leakage_columns_and_correct_size():
    hist = _toy_history()
    frame, _ = build_training_frame(hist)
    assert len(frame) == len(hist)
    assert list(frame[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert frame["result"].isin(["H", "D", "A"]).all()


def test_probabilities_are_valid_distributions():
    hist = _toy_history(20)
    frame, state = build_training_frame(hist)
    bundle = model_mod.train(frame)
    feats = state.features_before("Alpha", "Gamma", "2526")
    proba = model_mod.predict_proba(bundle, pd.DataFrame([feats]))
    assert proba.shape == (1, 3)
    assert abs(proba.sum() - 1.0) < 1e-6
    assert (proba >= 0).all() and (proba <= 1).all()


def test_critic_routes_back_on_invalid_probabilities():
    state = {
        "predictions": [{
            "home": "A", "away": "B", "status": "predicted",
            "p_home": 1.5, "p_draw": 0.2, "p_away": 0.1,
        }],
        "fixtures": [{"home": "A", "away": "B"}],
        "model_metrics": {},
    }
    out = critic_agent(state)
    assert out["critiques"], "critic must flag broken probabilities"
    assert "force_retrain" in out["critic_directives"]


def test_critic_passes_valid_run():
    state = {
        "predictions": [{
            "home": "A", "away": "B", "status": "predicted",
            "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
        }],
        "fixtures": [{"home": "A", "away": "B"}],
        "model_metrics": {"holdout": {"log_loss": 0.95}},
    }
    out = critic_agent(state)
    assert not out["critiques"]


def test_team_resolver_alias_and_fuzzy():
    r = TeamResolver(["Man United", "Tottenham", "Ath Madrid"])
    assert r.resolve("Manchester United") == "Man United"
    assert r.resolve("Tottenham Hotspur") == "Tottenham"
    assert r.resolve("Atlético Madrid") == "Ath Madrid"
    assert r.resolve("Nonexistent FC") is None


def test_normalize_strips_affixes_and_accents():
    assert normalize("FC Bayern München") == "bayern munchen"
    assert normalize("Brighton & Hove Albion") == "brighton and hove albion"

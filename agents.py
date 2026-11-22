"""Agent nodes for the prediction graph.

Each agent is a decision-making stage over shared run state, not a fixed
pipe: the scout chooses and degrades data sources, the analyst decides when
history is insufficient (abstain), the predictor decides whether to
re-train, and the critic validates outputs and can route the run back to
the predictor with corrective directives before anything is published.

No LLM is used: decisions are explicit rules over model and data state, and
every decision is recorded in the run report for auditability.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import data_sources
import model as model_mod
from config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    DATA_DIR,
    HOLDOUT_SEASON,
    LOGLOSS_WARNING,
    MAX_CRITIC_RETRIES,
    MIN_COVERAGE,
    MIN_TEAM_MATCHES,
    RETRAIN_MAX_AGE_HOURS,
)
from features import build_training_frame, fixture_features

TRAIN_PKL = DATA_DIR / "training_features.pkl"
STATE_PKL = DATA_DIR / "league_state.pkl"
FIXTURE_FEATURES_JSON = DATA_DIR / "fixture_features.json"


def _log(state: dict, msg: str) -> None:
    state.setdefault("report", []).append(msg)


# --------------------------------------------------------------------------
def scout_agent(state: dict) -> dict:
    """Fetch the day's fixtures; extend horizon or degrade gracefully."""
    day = dt.date.fromisoformat(state["run_date"])
    try:
        fixtures, date_used, notes = data_sources.fetch_fixtures(day)
    except data_sources.FetchError as exc:
        # source unreachable -> retry against the local cache before giving up
        try:
            fixtures, date_used, notes = data_sources.fetch_fixtures(day, use_cache=True)
            notes.append(f"live source failed ({exc}); used cached fixtures sheet")
        except data_sources.FetchError as exc2:
            state.update(fixtures=[], fixture_date=state["run_date"], scout_notes=[str(exc2)], degraded=True)
            _log(state, f"scout: fixture source unreachable: {exc2}")
            return state
    state["fixtures"] = [f.to_dict() for f in fixtures]
    state["fixture_date"] = date_used
    state["scout_notes"] = notes
    state["degraded"] = False
    _log(state, f"scout: {len(fixtures)} fixtures for {date_used} ({'; '.join(notes) or 'on target date'})")
    return state


# --------------------------------------------------------------------------
def analyst_agent(state: dict) -> dict:
    """Build history, replay team state, and decide data sufficiency."""
    history, errors = data_sources.fetch_history(use_cache=not state.get("no_cache", False))
    for e in errors:
        state.setdefault("warnings", []).append(f"history gap: {e}")
    frame, league_state = build_training_frame(history)
    frame.to_pickle(TRAIN_PKL)
    with STATE_PKL.open("wb") as fh:
        pickle.dump(league_state, fh)

    rows = []
    for fx in state.get("fixtures", []):
        feats = fixture_features(league_state, fx["home"], fx["away"], HOLDOUT_SEASON)
        sufficient = feats["home_matches"] >= MIN_TEAM_MATCHES and feats["away_matches"] >= MIN_TEAM_MATCHES
        row = {**fx, "features": feats, "sufficient_history": sufficient}
        if not sufficient:
            row["abstain_reason"] = (
                f"insufficient history (home {feats['home_matches']}, away {feats['away_matches']} matches)"
            )
        rows.append(row)
    FIXTURE_FEATURES_JSON.write_text(json.dumps(rows, indent=2, default=str))
    state["analyzed"] = len(rows)
    _log(state, f"analyst: {len(frame)} historical matches replayed, {len(rows)} fixtures featured")
    return state


# --------------------------------------------------------------------------
def predictor_agent(state: dict) -> dict:
    """Decide whether to re-train, then predict with confidence tiers."""
    directives = state.get("critic_directives", [])
    force_retrain = state.get("force_retrain", False) or "force_retrain" in directives

    bundle = model_mod.load()
    reasons = []
    if bundle is None:
        reasons.append("no saved model")
    else:
        age = model_mod.model_age_hours(bundle)
        if age > RETRAIN_MAX_AGE_HOURS:
            reasons.append(f"model age {age:.1f}h > {RETRAIN_MAX_AGE_HOURS}h")
    if force_retrain:
        reasons.append("forced retrain")

    if reasons:
        frame = pd.read_pickle(TRAIN_PKL)
        bundle = model_mod.train(frame)
        model_mod.save(bundle)
        state["retrained"] = True
        _log(state, f"predictor: retrained ({'; '.join(reasons)}), "
                    f"{bundle.metrics['train_matches']} train / {bundle.metrics['holdout_matches']} holdout")
    else:
        state["retrained"] = False
        _log(state, "predictor: reused fresh saved model")
    state["model_metrics"] = bundle.metrics

    rows = json.loads(FIXTURE_FEATURES_JSON.read_text())
    predictions = []
    valid = [r for r in rows if r.get("sufficient_history")]
    if valid:
        X = pd.DataFrame([r["features"] for r in valid])
        proba = model_mod.predict_proba(bundle, X)  # [H, D, A]
        for r, (p_h, p_d, p_a) in zip(valid, proba):
            conf = "high" if max(p_h, p_d, p_a) >= CONFIDENCE_HIGH else (
                "medium" if max(p_h, p_d, p_a) >= CONFIDENCE_MEDIUM else "low")
            pick = ["home", "draw", "away"][int(np.argmax([p_h, p_d, p_a]))]
            predictions.append({
                **{k: r[k] for k in ("league_code", "league_name", "date", "kickoff", "home", "away")},
                "p_home": round(float(p_h), 4),
                "p_draw": round(float(p_d), 4),
                "p_away": round(float(p_a), 4),
                "pick": pick,
                "confidence": conf,
                "factors": _factors(r["features"]),
                "status": "predicted",
            })
    for r in rows:
        if not r.get("sufficient_history"):
            predictions.append({
                **{k: r[k] for k in ("league_code", "league_name", "date", "kickoff", "home", "away")},
                "status": "abstained",
                "reason": r.get("abstain_reason", "insufficient data"),
            })
    state["predictions"] = predictions
    _log(state, f"predictor: {len(valid)} predicted, {len(predictions) - len(valid)} abstained")
    return state


def _factors(f: dict) -> list[str]:
    out = []
    ed = f["elo_diff"]
    if abs(ed) >= 40:
        side = "Home" if ed > 0 else "Away"
        out.append(f"{side} side rated {abs(ed):.0f} Elo points stronger (incl. home edge)")
    form_gap = f["home_form_pts"] - f["away_form_pts"]
    if abs(form_gap) >= 0.8:
        side = "Home" if form_gap > 0 else "Away"
        out.append(f"{side} form better over last 5 ({f['home_form_pts']:.1f} vs {f['away_form_pts']:.1f} pts/game)")
    attack = f["home_gf_avg"] - f["away_ga_avg"]
    if attack >= 0.5:
        out.append(f"Home attack ({f['home_gf_avg']:.1f} gf/g) vs away defense ({f['away_ga_avg']:.1f} ga/g)")
    elif attack <= -0.5:
        out.append(f"Away defense concedes less ({f['away_ga_avg']:.1f} ga/g) than home scores ({f['home_gf_avg']:.1f} gf/g)")
    if not out:
        out.append("Teams closely matched on rating, form and goal averages")
    return out


# --------------------------------------------------------------------------
def critic_agent(state: dict) -> dict:
    """Validate predictions; route back to the predictor when fixable."""
    critiques: list[str] = []
    directives: list[str] = []
    preds = state.get("predictions", [])
    predicted = [p for p in preds if p.get("status") == "predicted"]

    for p in predicted:
        s = p["p_home"] + p["p_draw"] + p["p_away"]
        vals = [p["p_home"], p["p_draw"], p["p_away"]]
        if not (0.99 <= s <= 1.01) or any(v < 0 or v > 1 for v in vals) or any(np.isnan(v) for v in vals):
            critiques.append(f"invalid probabilities for {p['home']} vs {p['away']}")
            directives.append("force_retrain")

    fixtures = state.get("fixtures", [])
    if fixtures:
        coverage = len(predicted) / len(fixtures)
        if coverage < MIN_COVERAGE:
            critiques.append(f"low coverage {coverage:.0%} of fixtures")
            # not fixable by retraining (data sufficiency), so warn instead
            state.setdefault("warnings", []).append(
                f"only {len(predicted)}/{len(fixtures)} fixtures had enough history to predict")

    metrics = state.get("model_metrics", {})
    hold = metrics.get("holdout") or {}
    if hold and hold.get("log_loss", 0) > LOGLOSS_WARNING:
        state.setdefault("warnings", []).append(
            f"holdout log-loss {hold['log_loss']} above {LOGLOSS_WARNING} threshold")

    state["critiques"] = critiques
    state["critic_directives"] = directives
    state["retries"] = state.get("retries", 0) + (1 if critiques else 0)
    if critiques:
        _log(state, f"critic: {len(critiques)} problem(s): {'; '.join(critiques)}")
    else:
        _log(state, "critic: all checks passed")
    return state


def route_after_critic(state: dict) -> str:
    if state.get("critiques") and state.get("retries", 0) < MAX_CRITIC_RETRIES:
        return "predictor"
    if state.get("critiques"):
        state.setdefault("warnings", []).append("critic issues unresolved after retries; publishing with warnings")
    return "pundit"


# --------------------------------------------------------------------------
def pundit_agent(state: dict) -> dict:
    """Plain-language run summary. Template-based; no LLM involved."""
    preds = state.get("predictions", [])
    predicted = [p for p in preds if p.get("status") == "predicted"]
    lines = []
    if predicted:
        best = max(predicted, key=lambda p: max(p["p_home"], p["p_draw"], p["p_away"]))
        pick_team = {"home": best["home"], "draw": "a draw", "away": best["away"]}[best["pick"]]
        prob = max(best["p_home"], best["p_draw"], best["p_away"])
        lines.append(
            f"{len(predicted)} matches predicted across {len({p['league_code'] for p in predicted})} leagues. "
            f"Strongest lean: {best['home']} vs {best['away']} - {pick_team} at {prob:.0%}.")
    else:
        lines.append("No predictions could be produced for this run.")
    if state.get("scout_notes"):
        lines.append(" / ".join(state["scout_notes"]))
    state["summary"] = lines
    _log(state, "pundit: summary written")
    return state


# --------------------------------------------------------------------------
def publisher_agent(state: dict) -> dict:
    from publish_site import render_site
    outputs = render_site(state)
    state["outputs"] = [str(p) for p in outputs]
    _log(state, f"publisher: wrote {len(outputs)} files to docs/")
    return state

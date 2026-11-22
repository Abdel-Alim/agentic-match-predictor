"""Point-in-time feature engineering.

Team state (Elo, recent form, goal averages) is built by replaying history
chronologically. For every match we snapshot the state *before* kickoff, so
training features never leak future results. The same machinery produces
features for upcoming fixtures.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

from config import (
    ELO_HOME_BONUS,
    ELO_K,
    ELO_NEW_TEAM,
    ELO_START,
    SEASON_REGRESSION,
)

FEATURE_COLUMNS = [
    "elo_diff",
    "home_form_pts",
    "away_form_pts",
    "home_gf_avg",
    "away_gf_avg",
    "home_ga_avg",
    "away_ga_avg",
    "home_matches",
    "away_matches",
]


@dataclass
class TeamState:
    elo: float = ELO_START
    seen: bool = False
    form: deque = field(default_factory=lambda: deque(maxlen=5))     # points per game
    gf: deque = field(default_factory=lambda: deque(maxlen=10))      # goals for
    ga: deque = field(default_factory=lambda: deque(maxlen=10))      # goals against

    @property
    def matches(self) -> int:
        return len(self.ga)  # ga deque fills every played game

    def snapshot(self) -> dict:
        return {
            "elo": self.elo,
            "form_pts": sum(self.form) / len(self.form) if self.form else 1.0,
            "gf_avg": sum(self.gf) / len(self.gf) if self.gf else 1.2,
            "ga_avg": sum(self.ga) / len(self.ga) if self.ga else 1.2,
            "matches": self.matches,
        }


class LeagueState:
    """Replays matches in order, keeping a per-team state."""

    def __init__(self):
        self.teams: dict[str, TeamState] = defaultdict(TeamState)
        self._current_season: str | None = None

    def _regress_for_new_season(self, season: str) -> None:
        if self._current_season == season:
            return
        if self._current_season is not None:
            for st in self.teams.values():
                st.elo = ELO_START + (st.elo - ELO_START) * SEASON_REGRESSION
        self._current_season = season

    def _team(self, name: str) -> TeamState:
        st = self.teams[name]
        if not st.seen:
            st.elo = ELO_NEW_TEAM if self._current_season != "first" else st.elo
            st.seen = True
        return st

    def features_before(self, home: str, away: str, season: str) -> dict:
        self._regress_for_new_season(season)
        h, a = self._team(home), self._team(away)
        hs, as_ = h.snapshot(), a.snapshot()
        return {
            "elo_diff": (hs["elo"] + ELO_HOME_BONUS) - as_["elo"],
            "home_form_pts": hs["form_pts"],
            "away_form_pts": as_["form_pts"],
            "home_gf_avg": hs["gf_avg"],
            "away_gf_avg": as_["gf_avg"],
            "home_ga_avg": hs["ga_avg"],
            "away_ga_avg": as_["ga_avg"],
            "home_matches": hs["matches"],
            "away_matches": as_["matches"],
        }

    def update(self, home: str, away: str, hg: int, ag: int, season: str) -> None:
        self._regress_for_new_season(season)
        h, a = self._team(home), self._team(away)
        expected_h = 1.0 / (1.0 + 10 ** (-((h.elo + ELO_HOME_BONUS) - a.elo) / 400.0))
        score_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = ELO_K * (score_h - expected_h)
        h.elo += delta
        a.elo -= delta
        h.form.append(3 if hg > ag else (1 if hg == ag else 0))
        a.form.append(3 if ag > hg else (1 if hg == ag else 0))
        h.gf.append(hg); h.ga.append(ag)
        a.gf.append(ag); a.ga.append(hg)


def build_training_frame(history: pd.DataFrame) -> tuple[pd.DataFrame, LeagueState]:
    """Replay all history; return per-match pre-game features + final state."""
    state = LeagueState()
    rows = []
    history = history.sort_values(["date"]).reset_index(drop=True)
    for _, m in history.iterrows():
        feats = state.features_before(m["home"], m["away"], m["season"])
        rows.append(feats)
        state.update(m["home"], m["away"], int(m["home_goals"]), int(m["away_goals"]), m["season"])
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    out = pd.concat([history.reset_index(drop=True), X], axis=1)
    return out, state


def fixture_features(state: LeagueState, home: str, away: str, season: str) -> dict:
    return state.features_before(home, away, season)

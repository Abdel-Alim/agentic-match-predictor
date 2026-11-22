"""Project configuration: leagues, data sources, paths, thresholds.

Everything environment-specific lives here so agents stay declarative.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"  # GitHub Pages serves from /docs on the default branch
for _d in (DATA_DIR, CACHE_DIR, MODEL_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TIMEZONE = os.environ.get("PREDICTOR_TIMEZONE", "Africa/Cairo")


@dataclass(frozen=True)
class League:
    code: str          # football-data.co.uk division code
    name: str
    country: str
    espn_slug: str     # ESPN scoreboard slug


LEAGUES: list[League] = [
    League("E0", "Premier League", "England", "eng.1"),
    League("SP1", "La Liga", "Spain", "esp.1"),
    League("I1", "Serie A", "Italy", "ita.1"),
    League("D1", "Bundesliga", "Germany", "ger.1"),
    League("F1", "Ligue 1", "France", "fra.1"),
]

# Seasons used for training. The last one doubles as the honest holdout:
# the model is trained on earlier seasons, then evaluated on the current one.
SEASONS: list[str] = ["2223", "2324", "2425", "2526"]
HOLDOUT_SEASON = "2526"

FD_CSV_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FD_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
)

HEADERS = {"User-Agent": "agentic-match-predictor/1.0 (personal research project)"}
HTTP_TIMEOUT = 25

# --- model / agent thresholds -------------------------------------------------
ELO_START = 1500.0
ELO_NEW_TEAM = 1400.0        # promoted / unseen teams start below average
ELO_K = 20.0
ELO_HOME_BONUS = 65.0
SEASON_REGRESSION = 0.70     # pull ratings toward the mean between seasons

MIN_TEAM_MATCHES = 5         # abstain below this much history
CONFIDENCE_HIGH = 0.55
CONFIDENCE_MEDIUM = 0.45
MAX_CRITIC_RETRIES = 2
MIN_COVERAGE = 0.60          # fraction of fixtures that must get predictions
LOGLOSS_WARNING = 1.05       # holdout log-loss above this triggers a warning
RETRAIN_MAX_AGE_HOURS = 20   # predictor re-trains when the model is older

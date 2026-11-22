# Agentic Match Predictor

An agentic AI pipeline that predicts football (soccer) match outcomes every day,
evaluates itself honestly, and publishes the results to a live website — fully
automated with GitHub Actions and GitHub Pages, at zero cost.

**Live predictions:** https://abdel-alim.github.io/agentic-match-predictor/

Leagues covered: Premier League, La Liga, Serie A, Bundesliga, Ligue 1.

---

## How it works

The pipeline is a [LangGraph](https://github.com/langchain-ai/langgraph) state
graph of specialized agents with a real feedback loop, not a fixed script:

```
scout ──> analyst ──> predictor ──> critic ──(fixable issue)──> predictor
                                    |
                                    └──────> pundit ──> publisher ──> END
```

| Agent | Decisions it makes |
|---|---|
| **Scout** | Fetches today's fixtures; if there are none, extends the horizon to the next matchday (within 5 days) and labels it. If the live source fails, falls back to cache; if that fails too, degrades gracefully instead of crashing. |
| **Analyst** | Replays ~7,000 historical matches chronologically to build point-in-time team state (Elo, form, goal averages) with zero future leakage. Decides which fixtures have enough history and marks the rest for abstention. |
| **Predictor** | Decides whether to re-train (model age, forced directives) or reuse the saved model. Produces calibrated 3-way probabilities, assigns confidence tiers, and abstains on low-data matches instead of guessing. |
| **Critic** | Validates every prediction (probabilities sum to 1, sane ranges, coverage, holdout quality). Fixable failures route the run **back** to the predictor with corrective directives; unfixable ones are published as explicit warnings. |
| **Pundit** | Writes the plain-language run summary from model factors (template-based, no LLM). |
| **Publisher** | Renders the static site and JSON feeds into `docs/`. |

Every decision is recorded in a run report (`docs/runs/<date>.json`).

## Model and features

- **Features** (all computed strictly before kickoff): Elo difference
  (with home advantage and between-season regression), points per game over the
  last 5, goals for/against averages over the last 10, matches played.
- **Model:** multinomial logistic regression (scikit-learn pipeline with
  feature scaling), retrained daily as new results land.
- **Output:** home / draw / away probabilities with confidence tiers and the
  key factors behind each pick.

## Honest evaluation

Metrics below are computed on the current season's completed matches that the
model **never trained on** (time-based holdout), and are regenerated every run:

| | Accuracy | Log-loss (lower = better) | Brier (lower = better) |
|---|---|---|---|
| This model | 0.511 | 1.000 | 0.597 |
| Baseline: always pick most common result | 0.440 | 1.072 | - |
| Baseline: bookmaker odds (market average) | 0.534 | 0.978 | - |

The model clearly beats naive baselines and approaches — but does not claim to
beat — the betting market. Probabilities are estimates, not guarantees, and
nothing here is betting advice.

## Data sources and licensing

All data comes from [football-data.co.uk](https://football-data.co.uk/data.php),
which publishes its compiled results and fixtures CSVs for download. The
pipeline downloads them at runtime; **no raw data is committed to this repo**
and the published site shows only fixture facts plus this project's own
computed probabilities and metrics. Attribution retained per the source's
request. The upstream fixtures sheet refreshes twice a week (Friday and
Tuesday), so very late fixture changes can lag one cycle.

## Automation

`.github/workflows/daily.yml` runs the graph every day at 05:30 UTC
(07:30 Cairo, winter) and on manual trigger: install, run, test, then commit
the updated site and model. GitHub Pages serves `docs/` from the default
branch.

## Run it locally

```bash
pip install -r requirements.txt
python main.py                    # today's predictions -> docs/
python main.py --date 2026-11-23  # a specific date
python -m pytest tests/ -q        # 7 pipeline tests
```

## Project structure

```
main.py            CLI entry point
graph.py           LangGraph topology (incl. the critic -> predictor retry loop)
agents.py          the six agent nodes
data_sources.py    source downloads, fixtures horizon logic, graceful failure
features.py        point-in-time Elo / form feature engine
model.py           training, honest holdout evaluation, baselines, persistence
teams.py           cross-source team-name resolution
publish_site.py    static site renderer (docs/)
tests/             pipeline tests
.github/workflows/ daily automation
docs/              generated website (GitHub Pages)
models/            trained model + metrics (regenerated daily)
```

## Roadmap

- Player-availability and odds-movement features
- Probability calibration plots on the site
- Accuracy tracking page per league over time

"""Data acquisition: historical results and upcoming fixtures.

Single owning source: football-data.co.uk, which publishes its compiled CSV
files explicitly for download and use (https://football-data.co.uk/data.php).
We download at runtime and never redistribute the raw data: the repo commits
no CSVs, and the published site shows only fixture facts (who plays whom,
when) plus this project's own computed probabilities and metrics.

Licensing posture: the fixtures/results CSVs are the site's own published
downloads. No unofficial or reverse-engineered endpoints are used.

Network errors never crash the pipeline: fetchers raise FetchError and the
scout agent decides how to route around the failure.
"""
from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass, field

import pandas as pd
import requests

from config import (
    CACHE_DIR,
    FD_CSV_URL,
    FD_FIXTURES_URL,
    HEADERS,
    HTTP_TIMEOUT,
    LEAGUES,
    SEASONS,
    League,
)


class FetchError(RuntimeError):
    pass


def _get(url: str) -> requests.Response:
    try:
        r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r
    except requests.RequestException as exc:  # network, DNS, HTTP status
        raise FetchError(f"{url}: {exc}") from exc


# --------------------------------------------------------------------------
# Historical results
# --------------------------------------------------------------------------

def fetch_fd_season(league: League, season: str, use_cache: bool = True) -> pd.DataFrame:
    """One league-season of results as a tidy DataFrame."""
    cache = CACHE_DIR / f"{league.code}_{season}.csv"
    if use_cache and cache.exists() and cache.stat().st_size > 0:
        raw = cache.read_bytes()
    else:
        raw = _get(FD_CSV_URL.format(season=season, code=league.code)).content
        cache.write_bytes(raw)
    df = pd.read_csv(io.StringIO(raw.decode("utf-8-sig", errors="replace")), on_bad_lines="skip")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]  # strip BOM from first header
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    df = df[df["FTR"].isin(["H", "D", "A"])]
    out = pd.DataFrame(
        {
            "league": league.code,
            "league_name": league.name,
            "season": season,
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "home": df["HomeTeam"].str.strip(),
            "away": df["AwayTeam"].str.strip(),
            "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
            "result": df["FTR"],
            "odd_home": pd.to_numeric(df.get("AvgH"), errors="coerce"),
            "odd_draw": pd.to_numeric(df.get("AvgD"), errors="coerce"),
            "odd_away": pd.to_numeric(df.get("AvgA"), errors="coerce"),
        }
    ).dropna(subset=["date", "home_goals", "away_goals"])
    return out.reset_index(drop=True)


def fetch_history(use_cache: bool = True) -> tuple[pd.DataFrame, list[str]]:
    frames, errors = [], []
    for lg in LEAGUES:
        for season in SEASONS:
            try:
                frames.append(fetch_fd_season(lg, season, use_cache=use_cache))
            except FetchError as exc:
                errors.append(f"{lg.code} {season}: {exc}")
    if not frames:
        raise FetchError("no historical data at all: " + "; ".join(errors))
    hist = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return hist, errors


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@dataclass
class Fixture:
    league_code: str
    league_name: str
    date: str                 # ISO date
    kickoff: str              # time as published by the source
    home: str
    away: str
    source: str = "football-data"
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "league_code": self.league_code,
            "league_name": self.league_name,
            "date": self.date,
            "kickoff": self.kickoff,
            "home": self.home,
            "away": self.away,
            "source": self.source,
            "issues": list(self.issues),
        }


def load_fixtures_table(use_cache: bool = False) -> pd.DataFrame:
    """The source's upcoming-fixtures sheet (updated twice weekly upstream)."""
    cache = CACHE_DIR / "fixtures.csv"
    if use_cache and cache.exists() and cache.stat().st_size > 0:
        raw = cache.read_bytes()
    else:
        raw = _get(FD_FIXTURES_URL).content
        cache.write_bytes(raw)
    df = pd.read_csv(io.StringIO(raw.decode("utf-8-sig", errors="replace")), on_bad_lines="skip")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]  # strip BOM from first header
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date", "Div"])
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date"])


def fetch_fixtures(
    day: dt.date,
    lookahead_days: int = 5,
    use_cache: bool = False,
) -> tuple[list[Fixture], str, list[str]]:
    """Fixtures for `day`; if none, the scout extends to the nearest matchday.

    Returns (fixtures, date_used_iso, notes).
    """
    codes = {lg.code: lg for lg in LEAGUES}
    notes: list[str] = []
    df = load_fixtures_table(use_cache=use_cache)
    df = df[df["Div"].isin(codes.keys())]
    horizon_end = day + dt.timedelta(days=lookahead_days)

    chosen = day
    day_df = df[df["date"].dt.date == day]
    if day_df.empty:
        future = df[(df["date"].dt.date > day) & (df["date"].dt.date <= horizon_end)]
        if future.empty:
            return [], day.isoformat(), ["لا توجد مباريات اليوم ولا ضمن نافذة البحث القادمة"]
        chosen = future["date"].min().date()
        day_df = df[df["date"].dt.date == chosen]
        notes.append(
            f"لا توجد مباريات يوم {day.isoformat()}؛ يتم عرض الجولة التالية {chosen.isoformat()}"
        )

    fixtures = [
        Fixture(
            league_code=row["Div"],
            league_name=codes[row["Div"]].name,
            date=chosen.isoformat(),
            kickoff=str(row.get("Time", "") or ""),
            home=str(row["HomeTeam"]).strip(),
            away=str(row["AwayTeam"]).strip(),
        )
        for _, row in day_df.iterrows()
    ]
    return fixtures, chosen.isoformat(), notes

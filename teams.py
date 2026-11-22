"""Team-name resolution between data sources.

football-data.co.uk uses short historical names ("Man United"),
ESPN uses display names ("Manchester United"). The resolver normalizes,
applies a curated alias table, then falls back to a fuzzy match and
finally reports the team as unknown so the pipeline can abstain instead
of guessing.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

_STRIP_PREFIXES = ("fc ", "afc ", "1. fc ", "sv ", "vfl ", "vfb ", "sc ", "rc ", "ac ", "as ", "ogc ", "tsg ", "rcd ", "ssc ")
_STRIP_SUFFIXES = (" fc", " cf", " afc")

_ALIASES = {
    # England
    "manchester united": "Man United",
    "manchester city": "Man City",
    "tottenham hotspur": "Tottenham",
    "tottenham": "Tottenham",
    "nottingham forest": "Nott'm Forest",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves",
    "brighton and hove albion": "Brighton",
    "newcastle united": "Newcastle",
    "west ham united": "West Ham",
    "leeds united": "Leeds",
    "bournemouth": "Bournemouth",
    # Spain
    "atletico madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao",
    "real sociedad": "Sociedad",
    "rayo vallecano": "Vallecano",
    "celta vigo": "Celta",
    "deportivo alaves": "Alaves",
    "alaves": "Alaves",
    "real betis": "Betis",
    "espanyol": "Espanol",
    "real oviedo": "Oviedo",
    "real madrid": "Real Madrid",
    # Italy
    "internazionale": "Inter",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "hellas verona": "Verona",
    "as roma": "Roma",
    "ss lazio": "Lazio",
    "acf fiorentina": "Fiorentina",
    # Germany
    "bayern munchen": "Bayern Munich",
    "bayern munich": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "borussia monchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "fc koln": "FC Koln",
    "fc cologne": "FC Koln",
    "1 fc koln": "FC Koln",
    "koln": "FC Koln",
    "vfl wolfsburg": "Wolfsburg",
    "wolfsburg": "Wolfsburg",
    "werder bremen": "Werder Bremen",
    "fc st pauli": "St Pauli",
    "st pauli": "St Pauli",
    "union berlin": "Union Berlin",
    "hamburger sv": "Hamburg",
    "hamburg sv": "Hamburg",
    "vfb stuttgart": "Stuttgart",
    "stuttgart": "Stuttgart",
    "sc freiburg": "Freiburg",
    "freiburg": "Freiburg",
    "tsg hoffenheim": "Hoffenheim",
    "hoffenheim": "Hoffenheim",
    "bayer leverkusen": "Leverkusen",
    "bayer 04 leverkusen": "Leverkusen",
    "leverkusen": "Leverkusen",
    "1 fc heidenheim": "Heidenheim",
    "heidenheim": "Heidenheim",
    "mainz 05": "Mainz",
    "mainz": "Mainz",
    "fc augsburg": "Augsburg",
    "augsburg": "Augsburg",
    "rb leipzig": "RB Leipzig",
    # France
    "paris saint germain": "Paris SG",
    "psg": "Paris SG",
    "paris sg": "Paris SG",
    "stade rennais": "Rennes",
    "rennes": "Rennes",
    "olympique marseille": "Marseille",
    "olympique de marseille": "Marseille",
    "olympique lyon": "Lyon",
    "olympique lyonnais": "Lyon",
    "as monaco": "Monaco",
    "monaco": "Monaco",
    "le havre ac": "Le Havre",
    "le havre": "Le Havre",
    "fc nantes": "Nantes",
    "ogc nice": "Nice",
    "rc lens": "Lens",
    "stade brestois": "Brest",
    "stade brestois 29": "Brest",
    "paris fc": "Paris FC",
}


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for p in _STRIP_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    for suf in _STRIP_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s.strip()


class TeamResolver:
    """Resolve external display names to canonical football-data names."""

    def __init__(self, canonical_names: list[str]):
        self.canonical = list(canonical_names)
        self._norm_to_canon = {normalize(c): c for c in self.canonical}

    def resolve(self, display_name: str, fuzzy_cutoff: float = 0.82) -> str | None:
        n = normalize(display_name)
        if n in _ALIASES:
            return _ALIASES[n]
        if n in self._norm_to_canon:
            return self._norm_to_canon[n]
        candidates = list(self._norm_to_canon.keys())
        close = difflib.get_close_matches(n, candidates, n=1, cutoff=fuzzy_cutoff)
        if close:
            return self._norm_to_canon[close[0]]
        return None

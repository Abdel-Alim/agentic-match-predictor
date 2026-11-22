"""Arabic display names for leagues and teams.

The pipeline keeps canonical football-data.co.uk names internally; only the
published site renders Arabic. Unknown names fall back to the canonical
English name so a newly promoted team never breaks rendering.
"""
from __future__ import annotations

LEAGUE_AR = {
    "Premier League": "الدوري الإنجليزي الممتاز",
    "La Liga": "الدوري الإسباني",
    "Serie A": "الدوري الإيطالي",
    "Bundesliga": "الدوري الألماني",
    "Ligue 1": "الدوري الفرنسي",
}

TEAM_AR = {
    # England
    "Arsenal": "أرسنال",
    "Aston Villa": "أستون فيلا",
    "Bournemouth": "بورنموث",
    "Brentford": "برينتفورد",
    "Brighton": "برايتون",
    "Burnley": "بيرنلي",
    "Chelsea": "تشيلسي",
    "Crystal Palace": "كريستال بالاس",
    "Everton": "إيفرتون",
    "Fulham": "فولهام",
    "Leeds": "ليدز يونايتد",
    "Liverpool": "ليفربول",
    "Man City": "مانشستر سيتي",
    "Man United": "مانشستر يونايتد",
    "Newcastle": "نيوكاسل يونايتد",
    "Nott'm Forest": "نوتينغهام فورست",
    "Sunderland": "سندرلاند",
    "Tottenham": "توتنهام هوتسبير",
    "West Ham": "وست هام يونايتد",
    "Wolves": "ولفرهامبتون",
    # Spain
    "Alaves": "ديبورتيفو ألافيس",
    "Ath Bilbao": "أتلتيك بلباو",
    "Ath Madrid": "أتلتيكو مدريد",
    "Barcelona": "برشلونة",
    "Betis": "ريال بيتيس",
    "Celta": "سيلتا فيغو",
    "Elche": "إلتشي",
    "Espanol": "إسبانيول",
    "Getafe": "خيتافي",
    "Girona": "جيرونا",
    "Levante": "ليفانتي",
    "Mallorca": "ريال مايوركا",
    "Osasuna": "أوساسونا",
    "Oviedo": "ريال أوفييدو",
    "Vallecano": "رايو فاليكانو",
    "Real Madrid": "ريال مدريد",
    "Sociedad": "ريال سوسييداد",
    "Sevilla": "إشبيلية",
    "Valencia": "فالنسيا",
    "Villarreal": "فياريال",
    # Italy
    "Atalanta": "أتالانتا",
    "Bologna": "بولونيا",
    "Cagliari": "كالياري",
    "Como": "كومو",
    "Cremonese": "كريمونيزي",
    "Fiorentina": "فيورنتينا",
    "Genoa": "جنوى",
    "Inter": "إنتر ميلان",
    "Juventus": "يوفنتوس",
    "Lazio": "لاتسيو",
    "Lecce": "ليتشي",
    "Milan": "إيه سي ميلان",
    "Monza": "مونزا",
    "Napoli": "نابولي",
    "Parma": "بارما",
    "Pisa": "بيزا",
    "Roma": "روما",
    "Sassuolo": "ساسوولو",
    "Torino": "تورينو",
    "Udinese": "أودينيزي",
    "Verona": "هيلاس فيرونا",
    # Germany
    "Augsburg": "أوغسبورغ",
    "Bayern Munich": "بايرن ميونخ",
    "Dortmund": "بوروسيا دورتموند",
    "Ein Frankfurt": "آينتراخت فرانكفورت",
    "Freiburg": "فرايبورغ",
    "FC Koln": "كولن",
    "Hamburg": "هامبورغ",
    "Heidenheim": "هايدنهايم",
    "Hoffenheim": "هوفنهايم",
    "Leverkusen": "باير ليفركوزن",
    "Mainz": "ماينتز",
    "M'gladbach": "بوروسيا مونشنغلادباخ",
    "RB Leipzig": "لايبزيغ",
    "St Pauli": "سانت باولي",
    "Stuttgart": "شتوتغارت",
    "Union Berlin": "يونيون برلين",
    "Werder Bremen": "فيردر بريمن",
    "Wolfsburg": "فولفسبورغ",
    # France
    "Angers": "أنجيه",
    "Auxerre": "أوكسير",
    "Brest": "بريست",
    "Le Havre": "لوهافر",
    "Lens": "لانس",
    "Lille": "ليل",
    "Lorient": "لوريان",
    "Lyon": "ليون",
    "Marseille": "مارسيليا",
    "Metz": "ميتز",
    "Monaco": "موناكو",
    "Nantes": "نانت",
    "Nice": "نيس",
    "Paris FC": "باريس إف سي",
    "Paris SG": "باريس سان جيرمان",
    "Rennes": "رين",
    "Strasbourg": "ستراسبورغ",
    "Toulouse": "تولوز",
}


def league_ar(name: str) -> str:
    return LEAGUE_AR.get(name, name)


def team_ar(name: str) -> str:
    return TEAM_AR.get(name, name)

"""Static site generation for GitHub Pages (docs/ folder)."""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

from config import DOCS_DIR, HOLDOUT_SEASON

CSS = """
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--fg:#e6edf3;--muted:#8b949e;--accent:#2f81f7;--home:#3fb950;--draw:#d29922;--away:#f85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.05rem;margin:28px 0 10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--muted);font-size:.9rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:10px 0}
.match{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between}
.teams{font-weight:600;font-size:1.02rem}.meta{color:var(--muted);font-size:.82rem}
.bars{display:flex;height:26px;border-radius:6px;overflow:hidden;margin:12px 0 6px;font-size:.78rem;font-weight:600;color:#0d1117}
.bars div{display:flex;align-items:center;justify-content:center;min-width:34px}
.bh{background:var(--home)}.bd{background:var(--draw)}.ba{background:var(--away)}
.legend{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:600;border:1px solid var(--border)}
.high{color:var(--home)}.medium{color:var(--draw)}.low{color:var(--away)}
.factors{margin:8px 0 0;padding-left:18px;color:var(--muted);font-size:.85rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border:1px solid var(--border);padding:6px 10px;text-align:left}
th{background:var(--card)}
.note{border-left:3px solid var(--accent);padding:8px 12px;background:var(--card);border-radius:0 8px 8px 0;margin:14px 0;font-size:.9rem}
.warn{border-left-color:var(--draw)}
footer{margin-top:36px;color:var(--muted);font-size:.8rem;line-height:1.6}
a{color:var(--accent)}
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _display_pct(p: dict) -> tuple[float, float, float]:
    """Round to whole percents that always sum to 100 (largest remainder)."""
    raw = [p["p_home"] * 100, p["p_draw"] * 100, p["p_away"] * 100]
    floored = [int(v) for v in raw]
    remainder = 100 - sum(floored)
    order = sorted(range(3), key=lambda i: raw[i] - floored[i], reverse=True)
    for i in order[:remainder]:
        floored[i] += 1
    return tuple(float(v) for v in floored)


def _match_card(p: dict) -> str:
    kickoff = str(p.get('kickoff', '')).strip()
    kick = _esc(p.get('date', '')) + (f" {_esc(kickoff)} UK time" if kickoff else "")
    head = f"""
    <div class="match">
      <div>
        <div class="teams">{_esc(p['home'])} <span class="meta">vs</span> {_esc(p['away'])}</div>
        <div class="meta">{_esc(p['league_name'])} &middot; {kick}</div>
      </div>
    </div>"""
    if p.get("status") == "abstained":
        return f'<div class="card">{head}<div class="meta">No prediction: {_esc(p.get("reason", "insufficient data"))}</div></div>'
    ph, pd_, pa = _display_pct(p)
    conf = _esc(p["confidence"])
    factors = "".join(f"<li>{_esc(f)}</li>" for f in p.get("factors", []))
    return f"""
  <div class="card">
    {head}
    <div class="bars">
      <div class="bh" style="width:{ph:.1f}%">{ph:.0f}%</div>
      <div class="bd" style="width:{pd_:.1f}%">{pd_:.0f}%</div>
      <div class="ba" style="width:{pa:.1f}%">{pa:.0f}%</div>
    </div>
    <div class="legend">
      <span>Home {ph:.0f}%</span><span>Draw {pd_:.0f}%</span><span>Away {pa:.0f}%</span>
      <span class="badge {conf}">{conf} confidence</span>
    </div>
    <ul class="factors">{factors}</ul>
  </div>"""


def _metrics_block(metrics: dict) -> str:
    hold = metrics.get("holdout")
    if not hold:
        return '<div class="note">Holdout evaluation not available yet (not enough completed matches this season).</div>'
    rows = [
        ("This model", hold.get("accuracy"), hold.get("log_loss"), hold.get("brier")),
    ]
    bm = metrics.get("baseline_majority", {})
    if bm:
        rows.append(("Baseline: always pick most common result", bm.get("accuracy"), bm.get("log_loss"), "-"))
    bb = metrics.get("baseline_bookmaker")
    if bb:
        rows.append(("Baseline: bookmaker odds (market average)", bb.get("accuracy"), bb.get("log_loss"), "-"))
    body = "".join(
        f"<tr><td>{_esc(n)}</td><td>{a}</td><td>{l}</td><td>{b}</td></tr>"
        for n, a, l, b in rows)
    return f"""
    <table>
      <tr><th>Evaluated on {metrics.get('holdout_matches')} completed {HOLDOUT_SEASON and 'current-season'} matches the model never trained on</th>
      <th>Accuracy</th><th>Log-loss (lower is better)</th><th>Brier (lower is better)</th></tr>
      {body}
    </table>"""


def render_site(state: dict) -> list[Path]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "runs").mkdir(exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    preds = state.get("predictions", [])
    metrics = state.get("model_metrics", {})
    warnings = state.get("warnings", [])
    summary = state.get("summary", [])

    by_league: dict[str, list[dict]] = {}
    for p in preds:
        by_league.setdefault(p["league_name"], []).append(p)

    sections = ""
    for league, items in by_league.items():
        cards = "".join(_match_card(p) for p in items)
        sections += f"<h2>{_esc(league)}</h2>{cards}"

    if not preds:
        sections = '<div class="card"><div class="meta">No fixtures available for this run. The source fixture sheet is updated twice a week; the next update is picked up automatically.</div></div>'

    warn_html = "".join(f'<div class="note warn">{_esc(w)}</div>' for w in warnings)
    summary_html = "".join(f'<div class="note">{_esc(s)}</div>' for s in summary)

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic Match Predictor - { _esc(state.get('fixture_date', '')) }</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>Agentic Match Predictor</h1>
<div class="sub">Predictions for { _esc(state.get('fixture_date', state.get('run_date',''))) } &middot;
generated {now.strftime('%Y-%m-%d %H:%M')} UTC by an automated agent pipeline</div>
{summary_html}
{warn_html}
{sections}
<h2>Model evaluation (honest, out-of-sample)</h2>
{_metrics_block(metrics)}
<div class="meta" style="margin-top:8px">Trained on {metrics.get('train_matches', 0)} completed matches.
Probabilities are model estimates, not guarantees. Nothing here is betting advice.</div>
<footer>
Data: <a href="https://www.football-data.co.uk/data.php">football-data.co.uk</a> (published CSV downloads; fixture list refreshes twice weekly at source).<br>
Pipeline: LangGraph agent graph (scout &rarr; analyst &rarr; predictor &rarr; critic &rarr; pundit &rarr; publisher), re-run daily by GitHub Actions. Match factors and summaries are template-generated from model features; no LLM writes them.<br>
Source code and methodology: see the repository README.
</footer>
</div></body></html>"""

    index = DOCS_DIR / "index.html"
    index.write_text(page)

    payload = {
        "generated_at": now.isoformat(),
        "run_date": state.get("run_date"),
        "fixture_date": state.get("fixture_date"),
        "predictions": preds,
        "warnings": warnings,
        "summary": summary,
    }
    (DOCS_DIR / "predictions.json").write_text(json.dumps(payload, indent=2))
    (DOCS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run_file = DOCS_DIR / "runs" / f"{state.get('run_date', 'unknown')}.json"
    run_file.write_text(json.dumps({**payload, "report": state.get("report", [])}, indent=2))
    return [index, DOCS_DIR / "predictions.json", DOCS_DIR / "metrics.json", run_file]

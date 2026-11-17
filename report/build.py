"""Build the static results page from results.json (ADR-005, US-14).

Fails with MissingSectionData if any publish-standard section has no data. No network, no JS
framework: one HTML file plus a copy of the validated results.json.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from evalkit.schema import validate

HERE = Path(__file__).resolve().parent
CONTENT = HERE / "content"

REQUIRED_SECTIONS = [
    "headline",
    "architecture",
    "tradeoffs",
    "benchmark",
    "losses",
    "failure-cases",
    "cost",
    "when-not-to-use",
    "repro",
]
FULL_MODE_STUDIES = ["leak_study", "controller_study", "time_study", "stacking_ablation"]

SERIES = {  # categorical slots 1-3 for the AutoML systems; floors are neutral with distinct shapes
    "tab": ("--series-1", "circle"),
    "flaml": ("--series-2", "circle"),
    "autogluon": ("--series-3", "circle"),
}
NEUTRAL_SHAPES = {"hgb_default": "square", "logreg": "triangle", "prior": "diamond"}


class MissingSectionData(ValueError):
    pass


# ------------------------------------------------------------------------------------------------
def md(text: str) -> str:
    """Tiny markdown subset: ### headings, - bullets, 4-space code blocks, `code`, **bold**."""
    out: list[str] = []
    in_list = in_code = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("    ") and not in_list:
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            out.append(html.escape(line[4:]))
            continue
        if in_code:
            out.append("</code></pre>")
            in_code = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out).replace("<pre><code>\n", "<pre><code>")


def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)


def _content(name: str) -> str:
    p = CONTENT / f"{name}.md"
    if not p.exists() or not p.read_text(encoding="utf-8").strip():
        raise MissingSectionData(f"section content {p.name} is missing or empty")
    return md(p.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        if math.isnan(x):
            return "–"
        return f"{x:.{nd}f}"
    return html.escape(str(x))


def _table(headers: list[str], rows: list[list[Any]], cls: str = "") -> str:
    th = "".join(f"<th scope='col'>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{c if isinstance(c, str) and c.startswith('<') else _fmt(c)}</td>" for c in r
        )
        + "</tr>"
        for r in rows
    )
    return f"<div class='table-wrap'><table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


# ------------------------------------------------------------------------------------------------
def _diagram() -> str:
    boxes = [
        ("bench/datasets", "fold → train.parquet + test_X.parquet", 20),
        ("runner", "spawn · time · memory (tree)", 200),
        ("system subprocess", "Tab / FLAML / AutoGluon / floors", 380),
        ("evalkit", "scores from proba.npy + withheld labels", 560),
        ("results.json → page", "schema-validated", 740),
    ]
    parts = []
    for i, (t, sub, x) in enumerate(boxes):
        parts.append(
            f"<g><rect x='{x}' y='30' width='160' height='70' rx='8' class='box'/>"
            f"<text x='{x + 80}' y='58' class='box-title'>{html.escape(t)}</text>"
            f"<text x='{x + 80}' y='80' class='box-sub'>{html.escape(sub)}</text></g>"
        )
        if i < len(boxes) - 1:
            parts.append(
                f"<line x1='{x + 162}' y1='65' x2='{x + 198}' y2='65' class='arrow' marker-end='url(#ah)'/>"
            )
    parts.append(
        "<text x='460' y='130' class='box-sub'>labels never cross into the system process; a system cannot write a score (schema rejects it)</text>"
    )
    return (
        "<figure class='diagram'><svg viewBox='0 0 920 145' role='img' aria-label='Data flow: datasets to runner to system subprocess to evalkit to results page'>"
        "<defs><marker id='ah' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' class='arrowhead'/></marker></defs>"
        + "".join(parts)
        + "</svg></figure>"
    )


def _marker(
    shape: str, x: float, y: float, cls_style: str, title: str, hit_area: bool = True
) -> str:
    t = f"<title>{html.escape(title)}</title>"
    hit = f"<circle cx='{x:.1f}' cy='{y:.1f}' r='10' class='hit'>{t}</circle>"
    if shape == "circle":
        m = f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' style='{cls_style}' class='pt'/>"
    elif shape == "square":
        m = f"<rect x='{x - 4.5:.1f}' y='{y - 4.5:.1f}' width='9' height='9' rx='1.5' style='{cls_style}' class='pt'/>"
    elif shape == "triangle":
        m = f"<path d='M{x:.1f},{y - 6:.1f} L{x + 5.5:.1f},{y + 4:.1f} L{x - 5.5:.1f},{y + 4:.1f} Z' style='{cls_style}' class='pt'/>"
    else:
        m = f"<path d='M{x:.1f},{y - 6:.1f} L{x + 6:.1f},{y:.1f} L{x:.1f},{y + 6:.1f} L{x - 6:.1f},{y:.1f} Z' style='{cls_style}' class='pt'/>"
    return m + hit if hit_area else m


def _scatter(per_dataset: list[dict[str, Any]]) -> str:
    """Primary plot (PREREG section 2): mean skill vs median actual seconds, log x."""
    pts = [r for r in per_dataset if r.get("median_actual_s") and r.get("mean_skill") is not None]
    if not pts:
        return ""
    W, H, L, R, T, B = 760, 360, 56, 20, 16, 44
    xs = [math.log10(max(r["median_actual_s"], 1e-2)) for r in pts]
    ys = [r["mean_skill"] for r in pts]
    x0, x1 = math.floor(min(xs)), math.ceil(max(xs) + 1e-9)
    if x1 == x0:
        x1 = x0 + 1
    y0 = min(0.0, math.floor(min(ys) * 10) / 10)
    y1 = max(0.1, math.ceil(max(ys) * 10) / 10)

    def sx(v: float) -> float:
        return L + (v - x0) / (x1 - x0) * (W - L - R)

    def sy(v: float) -> float:
        return T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)

    g = []
    for d in range(x0, x1 + 1):
        g.append(
            f"<line x1='{sx(d):.1f}' x2='{sx(d):.1f}' y1='{T}' y2='{H - B}' class='grid'/>"
            f"<text x='{sx(d):.1f}' y='{H - B + 18}' class='tick' text-anchor='middle'>{10**d:g} s</text>"
        )
    steps = 5
    for i in range(steps + 1):
        v = y0 + (y1 - y0) * i / steps
        g.append(
            f"<line x1='{L}' x2='{W - R}' y1='{sy(v):.1f}' y2='{sy(v):.1f}' class='grid{' zero' if abs(v) < 1e-12 else ''}'/>"
            f"<text x='{L - 8}' y='{sy(v) + 4:.1f}' class='tick' text-anchor='end'>{v:.1f}</text>"
        )
    marks = []
    for r, xv, yv in zip(pts, xs, ys, strict=True):
        var, shape = SERIES.get(
            r["system"], ("--neutral", NEUTRAL_SHAPES.get(r["system"], "diamond"))
        )
        title = f"{r['system']} · {r['dataset']}: skill {r['mean_skill']:.3f}, {r['median_actual_s']:.1f} s"
        marks.append(_marker(shape, sx(xv), sy(yv), f"fill:var({var})", title))
    systems = sorted({r["system"] for r in pts}, key=lambda s: (s not in SERIES, s))
    legend = "".join(
        "<span class='lg'><svg width='14' height='14' viewBox='0 0 14 14'>"
        + _marker(
            SERIES[s][1] if s in SERIES else NEUTRAL_SHAPES.get(s, "diamond"),
            7,
            7,
            f"fill:var({SERIES[s][0] if s in SERIES else '--neutral'})",
            s,
            hit_area=False,
        )
        + f"</svg>{html.escape(s)}</span>"
        for s in systems
    )
    return (
        "<figure class='chart'><figcaption>Skill score (log loss) versus actual seconds, one mark per system × dataset, mean over folds. "
        "Higher is better; left is cheaper. Hover a mark for its values; the table below has every number.</figcaption>"
        f"<div class='legend'>{legend}</div>"
        f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Scatter of skill score against actual seconds per system and dataset'>"
        + "".join(g)
        + f"<text x='{(L + W - R) / 2}' y='{H - 6}' class='axis-label' text-anchor='middle'>actual seconds (log scale, measured by the runner)</text>"
        + f"<text x='14' y='{(T + H - B) / 2}' class='axis-label' text-anchor='middle' transform='rotate(-90 14 {(T + H - B) / 2})'>skill score</text>"
        + "".join(marks)
        + "</svg></figure>"
    )


# ------------------------------------------------------------------------------------------------
def check_sections(doc: dict[str, Any]) -> None:
    if not doc["headline"].get("statement"):
        raise MissingSectionData("headline: no statement")
    if not doc["runs"]:
        raise MissingSectionData("benchmark: no runs")
    an = doc["analyses"]
    if not an.get("per_dataset"):
        raise MissingSectionData("losses: analyses.per_dataset is empty")
    if not an.get("cost"):
        raise MissingSectionData("cost: analyses.cost is empty")
    for name in ("architecture", "tradeoffs", "failure_modes", "when_not_to_use", "repro"):
        _content(name)
    if doc["config"]["mode"] == "full":
        for s in FULL_MODE_STUDIES:
            if not an.get(s):
                raise MissingSectionData(f"full mode needs analyses.{s}")


def _study_block(an: dict[str, Any]) -> str:
    blocks = []
    for key, title in [
        ("controller_study", "Budget controller study (pre-registered)"),
        ("leak_study", "Leakage detector study"),
        ("time_study", "Time-aware study"),
        ("stacking_ablation", "Stacking ablation"),
        ("floor_check", "Floor check"),
        ("memory", "Memory (NFR-3)"),
    ]:
        if an.get(key):
            v = an[key]
            summary = v.get("statement") or v.get("summary") or ""
            blocks.append(
                f"<h3>{html.escape(title)}</h3>"
                + (f"<p>{html.escape(summary)}</p>" if summary else "")
                + f"<details><summary>Data</summary><pre><code>{html.escape(json.dumps(v, indent=2)[:20000])}</code></pre></details>"
            )
    if not blocks:
        return "<p class='muted'>No studies in this run mode. The full benchmark adds the controller, leakage, time-aware and stacking studies.</p>"
    return "".join(blocks)


def render(doc: dict[str, Any]) -> str:
    an = doc["analyses"]
    head = doc["headline"]
    hw, cfg = doc["hardware"], doc["config"]
    claim_badge = (
        "<span class='badge claim'>claim</span>"
        if head["is_claim"]
        else "<span class='badge'>not yet a claim</span>"
    )
    comp_rows = [
        [
            c["baseline"],
            c["n_datasets"],
            c["mean_diff"],
            c["ci_low"],
            c["ci_high"],
            c["verdict"],
            f"{c['wins']}/{c['ties']}/{c['losses']}",
            (c.get("secondary") or {}).get("auc_mean_diff"),
        ]
        for c in head["comparisons"]
    ]
    per = an["per_dataset"]
    bench_rows = [
        [
            r["system"],
            r["dataset"],
            r.get("n_folds"),
            r.get("mean_skill"),
            r.get("mean_log_loss"),
            r.get("mean_auc"),
            r.get("median_actual_s"),
            r.get("max_peak_rss_gb"),
            ", ".join(r.get("failures") or []) or "–",
        ]
        for r in per
    ]
    losses = [
        r for r in per if r.get("vs_primary") and r["vs_primary"]["verdict"] in {"loss", "tie"}
    ]
    loss_rows = [
        [
            r["dataset"],
            r["system"],
            r["vs_primary"]["mean_diff"],
            r["vs_primary"]["ci_low"],
            r["vs_primary"]["ci_high"],
            r["vs_primary"]["verdict"],
        ]
        for r in losses
    ]
    failed = [r for r in doc["runs"] if r["status"] != "ok"]
    fail_rows = [
        [r["run_id"], r["status"], (r["error"] or "")[:160], r["actual_wall_s"]] for r in failed
    ]
    cost_rows = [
        [
            c.get("system"),
            c.get("n_runs"),
            c.get("median_actual_s"),
            c.get("p95_actual_s"),
            c.get("median_budget_ratio"),
            c.get("max_peak_rss_gb"),
            c.get("n_failed"),
        ]
        for c in an["cost"]
    ]
    nfr1 = an["nfr1"]
    nfr1_txt = (
        f"Tab finished within 1.10 × the ceiling on {nfr1['n_within']} of {nfr1['n_runs']} runs "
        f"({_fmt(nfr1['fraction_within'], 3)}; target ≥ 0.95: {'pass' if nfr1['passes'] else 'FAIL' if nfr1['passes'] is not None else 'n/a'})."
    )
    over = "".join(
        f"<li>{html.escape(o['run_id'])}: {o['budget_ratio']:.2f} × ceiling</li>"
        for o in nfr1["overruns"]
    )
    prov = (
        (
            "<p class='warn'>Tuned parameters are provisional (prereg/tuned_params.json not committed yet); "
            "these numbers are not pre-registered results.</p>"
        )
        if cfg.get("provisional_params")
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tab Results</title>
<style>{CSS}</style></head>
<body><main>
<header><p class="eyebrow">3A · Tab · tabular AutoML measured honestly</p>
<h1>Tab results</h1>
<p class="meta">mode <code>{html.escape(cfg["mode"])}</code> · git <code>{html.escape(doc["git_sha"][:12])}</code> · {html.escape(doc["created_utc"])} ·
<a href="results.json">results.json</a></p>{prov}</header>

<section id="headline"><h2>Headline {claim_badge}</h2>
<p class="lede">{html.escape(head["statement"])}</p>
<p class="muted">Metric: skill score on log loss (1 − model / class-prior log loss). AUC is shown for context and cannot change a verdict.</p>
{_table(["baseline", "datasets", "mean Δ skill", "CI low", "CI high", "verdict", "better/tied/worse", "AUC Δ (secondary)"], comp_rows) if comp_rows else "<p class='muted'>No baselines compared in this run.</p>"}
</section>

<section id="architecture"><h2>Problem and architecture</h2>{_diagram()}{_content("architecture")}</section>

<section id="tradeoffs"><h2>Design tradeoffs we rejected</h2>{_content("tradeoffs")}</section>

<section id="benchmark"><h2>Benchmark</h2>
<p>Methodology: every system runs as its own subprocess with the same ceiling (<strong>{cfg["budget_s"]:g} s</strong>), the same CPU count
({cfg.get("n_cpus", "–")}), folds {html.escape(str(cfg["folds"]))}, on {len(cfg["datasets"])} dataset(s). Wall-clock and memory are measured by the runner,
scores by evalkit from saved probabilities. Failed runs stay in the table and count as skill 0.</p>
<p class="muted">Hardware: {html.escape(hw["cpu_model"])}, {hw["n_logical_cpus"]} logical CPUs, {hw["ram_gb"]} GB RAM, {html.escape(hw["os"])}, Python {html.escape(hw["python"])}.
Versions: {html.escape(", ".join(f"{k} {v}" for k, v in doc["versions"].items()))}. Dataset list sha256 <code>{doc["dataset_list_sha256"][:16]}…</code></p>
{_scatter(per)}
{_table(["system", "dataset", "folds", "skill", "log loss", "AUC", "median s", "peak GB", "failures"], bench_rows)}
</section>

<section id="losses"><h2>Where Tab loses or ties</h2>
{_table(["dataset", "vs system", "mean Δ skill", "CI low", "CI high", "verdict"], loss_rows) if loss_rows else "<p>No per-dataset losses or ties against a compared system in this run (see the full table above).</p>"}
</section>

<section id="failure-cases"><h2>Failure cases</h2>{_content("failure_modes")}
<h3>Runs that failed in this benchmark</h3>
{_table(["run", "status", "error", "actual s"], fail_rows) if fail_rows else "<p>None: every run produced valid predictions.</p>"}
</section>

<section id="cost"><h2>Cost and latency</h2><p>{html.escape(nfr1_txt)}</p>{f"<ul>{over}</ul>" if over else ""}
{_table(["system", "runs", "median s", "p95 s", "median s / ceiling", "max peak GB (tree)", "failed"], cost_rows)}
<h3>Studies and guardrails</h3>{_study_block(an)}
</section>

<section id="when-not-to-use"><h2>When not to use Tab</h2>{_content("when_not_to_use")}</section>

<section id="repro"><h2>Reproduce it</h2>{_content("repro")}</section>
</main></body></html>
"""


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--surface:#ffffff;--text:#0b0b0b;--text-2:#52514e;--muted:#6f6e69;--line:#e4e3df;
--accent:#2a78d6;--warn:#8a5a00;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--neutral:#8a8984;--code:#f3f2ef}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#141413;--surface:#1a1a19;--text:#f0efec;
--text-2:#c3c2b7;--muted:#9d9c94;--line:#33332f;--accent:#3987e5;--warn:#e0b04a;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--neutral:#8f8e86;--code:#232321}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#141413;--surface:#1a1a19;--text:#f0efec;--text-2:#c3c2b7;--muted:#9d9c94;--line:#33332f;
--accent:#3987e5;--warn:#e0b04a;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--neutral:#8f8e86;--code:#232321}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}h1{font-size:2rem;margin:.2em 0}h2{font-size:1.35rem;margin:0 0 .6em}
h3{font-size:1.05rem;margin:1.4em 0 .4em}section{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}
.eyebrow{color:var(--muted);font-size:.85rem;letter-spacing:.04em;text-transform:uppercase;margin:0}.meta,.muted{color:var(--muted);font-size:.9rem}
.lede{font-size:1.1rem}.warn{color:var(--warn);font-weight:600}a{color:var(--accent)}code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em}
pre{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto}pre code{padding:0;background:none}
.badge{display:inline-block;font-size:.75rem;font-weight:600;border:1px solid var(--line);border-radius:999px;padding:2px 10px;margin-left:8px;color:var(--text-2);vertical-align:middle}
.badge.claim{border-color:var(--accent);color:var(--accent)}
.table-wrap{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--text-2);font-weight:600}
figure{margin:12px 0}figcaption{color:var(--text-2);font-size:.9rem;margin-bottom:6px}svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}.grid.zero{stroke:var(--muted)}.tick,.axis-label{fill:var(--muted);font-size:11px}
.pt{stroke:var(--surface);stroke-width:2}.hit{fill:transparent;cursor:default}.hit:hover{fill:var(--text);fill-opacity:.08}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.85rem;color:var(--text-2);margin:4px 0 8px}.lg{display:inline-flex;align-items:center;gap:6px}.lg svg{width:14px;height:14px}
.diagram .box{fill:var(--bg);stroke:var(--line)}.box-title{fill:var(--text);font-size:12px;font-weight:600;text-anchor:middle}
.box-sub{fill:var(--muted);font-size:9.5px;text-anchor:middle}.arrow{stroke:var(--muted);stroke-width:1.5}.arrowhead{fill:var(--muted)}
details summary{cursor:pointer;color:var(--text-2)}
"""


def build_page(doc: dict[str, Any], out_dir: Path) -> Path:
    validate(doc, "results")
    check_sections(doc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(render(doc), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the static results page")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "site")
    a = ap.parse_args(argv)
    doc = json.loads(a.results.read_text(encoding="utf-8"))
    if a.out.exists():
        shutil.rmtree(a.out)
    build_page(doc, a.out)
    print(f"page written to {a.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

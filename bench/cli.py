"""One entry point for every pipeline (Makefile targets are thin wrappers; works on native Windows).

python -m bench.cli skeleton            walking skeleton (fixture or --dataset credit-g)
python -m bench.cli repro               US-13 smoke repro, checked against ci/smoke_reference.json
python -m bench.cli repro-full          frozen tier 1 x 5 folds, every system, plus studies
python -m bench.cli verify              recompute every score from runs/ and compare
python -m bench.cli freeze              lock bench/selected_datasets.json (HF-2)
python -m bench.cli page --results F    build the static page
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bench import assemble as asm
from bench import datasets, paths, runner
from report.build import build_page

SMOKE_DATASETS = ["fixture_binary", "fixture_multiclass"]
SMOKE_SYSTEMS = ["prior", "logreg", "tab"]
SMOKE_BUDGET = 20.0
SMOKE_REFERENCE = paths.ROOT / "ci" / "smoke_reference.json"
FULL_SYSTEMS = ["tab", "flaml", "autogluon", "hgb_default", "logreg", "prior"]
FULL_BUDGET = 60.0


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_grid(systems, dsets, folds, budget, cpus, seed, runs_root, mem_limit_gb=None) -> list[dict]:
    recs = []
    for d in dsets:
        ds = datasets.load(d)
        for f in folds:
            sp = datasets.write_split(ds, f)
            for s in systems:
                r = runner.run_one(
                    s,
                    ds.name,
                    ds.task_id,
                    f,
                    seed,
                    budget,
                    cpus,
                    sp,
                    runs_root,
                    mem_limit_gb=mem_limit_gb,
                )
                sc = r["scores"]["skill"] if r["scores"] else None
                _log(
                    f"{s:12s} {d:28s} f{f}  {r['status']:7s} {r['actual_wall_s']:7.1f}s  skill={sc if sc is None else round(sc, 4)}"
                )
                recs.append(r)
    return recs


def check_reference(doc: dict, ref_path: Path = SMOKE_REFERENCE) -> list[str]:
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    problems = []
    for r in doc["runs"]:
        key = f"{r['system']}/{r['dataset']}"
        if key not in ref["skill_intervals"]:
            problems.append(f"{key}: no reference interval")
            continue
        lo, hi = ref["skill_intervals"][key]
        if r["status"] != "ok":
            problems.append(f"{key}: status {r['status']} ({r['error']})")
        elif not lo <= r["scores"]["skill"] <= hi:
            problems.append(
                f"{key}: skill {r['scores']['skill']:.4f} outside reference [{lo}, {hi}]"
            )
    return problems


def cmd_skeleton(a) -> int:
    runs_root = Path(a.runs)
    dset = a.dataset
    run_grid(["prior", "logreg"], [dset], [0], a.budget, a.cpus, 0, runs_root)
    out = paths.RESULTS / "skeleton" / "results.json"
    doc = asm.assemble(
        runs_root, out, "skeleton", a.budget, ["prior", "logreg"], [dset], [0], cpus=a.cpus
    )
    build_page(doc, paths.ROOT / "report" / "site" / "skeleton")
    _log(f"skeleton ok: {out}")
    return 0


def cmd_repro(a) -> int:
    t0 = time.perf_counter()
    runs_root = Path(a.runs)
    run_grid(SMOKE_SYSTEMS, SMOKE_DATASETS, [0], SMOKE_BUDGET, a.cpus, 0, runs_root)
    out = paths.RESULTS / "smoke" / "results.json"
    doc = asm.assemble(
        runs_root, out, "smoke", SMOKE_BUDGET, SMOKE_SYSTEMS, SMOKE_DATASETS, [0], cpus=a.cpus
    )
    build_page(doc, paths.ROOT / "report" / "site" / "smoke")
    problems = check_reference(doc)
    took = time.perf_counter() - t0
    _log(f"smoke repro took {took:.0f} s (NFR-7 limit 900 s)")
    if took > 900:
        problems.append(f"NFR-7: smoke repro took {took:.0f} s > 900 s")
    for p in problems:
        _log(f"FAIL {p}")
    if problems:
        return 1
    _log("smoke repro PASSED: every score inside its committed reference interval")
    return 0


def cmd_full(a) -> int:
    from bench import ablation, controller_study, leak_study, time_study

    if not datasets.check_frozen():
        _log(
            "refusing: bench/selected_datasets.json is not frozen (run `python -m bench.cli freeze` after pasting the list)"
        )
        return 2
    tuned = paths.ROOT / "prereg" / "tuned_params.json"
    if not tuned.exists() and not a.provisional:
        _log(
            "refusing: prereg/tuned_params.json missing. Run `python -m bench.tune_dev` and commit it first, or pass --provisional"
        )
        return 2
    sel = datasets.selected_list()
    tier1 = [e["name"] for e in sel["tier1"]]
    folds = sel.get("folds", [0, 1, 2, 3, 4])
    runs_root = Path(a.runs)
    systems = a.systems.split(",") if a.systems else FULL_SYSTEMS
    run_grid(systems, tier1, folds, FULL_BUDGET, a.cpus, 0, runs_root, mem_limit_gb=a.mem_limit_gb)
    studies = {}
    if not a.skip_studies:
        studies["leak_study"] = leak_study.run([e["name"] for e in sel["tier1"]])
        studies["stacking_ablation"] = ablation.run(tier1, folds, FULL_BUDGET, a.cpus, runs_root)
        studies["controller_study"] = controller_study.run(
            [e["name"] for e in sel["tier2"]], folds, a.cpus, runs_root
        )
        studies["time_study"] = time_study.run(a.cpus)
    out = paths.RESULTS / "full" / "results.json"
    doc = asm.assemble(
        runs_root, out, "full", FULL_BUDGET, systems, tier1, folds, cpus=a.cpus, studies=studies
    )
    build_page(doc, paths.ROOT / "report" / "site" / "full")
    _log(f"full benchmark assembled: {out}")
    return 0


def cmd_verify(a) -> int:
    from bench.verify import verify

    problems = verify(Path(a.results), Path(a.runs))
    for p in problems:
        _log(f"FAIL {p}")
    _log("verify PASSED" if not problems else f"verify FAILED ({len(problems)} problems)")
    return 1 if problems else 0


def cmd_freeze(a) -> int:
    sel = datasets.selected_list()
    if len(sel.get("tier1", [])) != 20 or len(sel.get("tier2", [])) != 8:
        _log("refusing: tier1 must list 20 tasks and tier2 8 tasks")
        return 2
    if paths.SELECTED_SHA.exists():
        _log("already frozen; changing the list after freeze violates HF-2")
        return 2
    sel["status"] = "frozen"
    paths.SELECTED.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
    paths.SELECTED_SHA.write_text(datasets.file_sha256(paths.SELECTED) + "\n", encoding="utf-8")
    _log(f"frozen: {paths.SELECTED_SHA.read_text().strip()} — commit both files now")
    return 0


def cmd_page(a) -> int:
    doc = json.loads(Path(a.results).read_text(encoding="utf-8"))
    build_page(doc, Path(a.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("skeleton", "repro", "repro-full", "verify", "freeze", "page"):
        p = sub.add_parser(name)
        p.add_argument("--runs", default=str(paths.RUNS))
        p.add_argument("--cpus", type=int, default=1)
        if name == "skeleton":
            p.add_argument("--dataset", default="fixture_binary")
            p.add_argument("--budget", type=float, default=20.0)
        if name == "repro-full":
            p.add_argument("--systems", default="")
            p.add_argument("--provisional", action="store_true")
            p.add_argument("--skip-studies", action="store_true")
            p.add_argument("--mem-limit-gb", type=float, default=12.0)
        if name in ("verify", "page"):
            p.add_argument("--results", required=True)
        if name == "page":
            p.add_argument("--out", default=str(paths.ROOT / "report" / "site"))
    a = ap.parse_args(argv)
    fn = {
        "skeleton": cmd_skeleton,
        "repro": cmd_repro,
        "repro-full": cmd_full,
        "verify": cmd_verify,
        "freeze": cmd_freeze,
        "page": cmd_page,
    }[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())

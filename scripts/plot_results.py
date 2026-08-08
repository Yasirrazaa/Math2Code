"""Generate the README benchmark figures from measured results.

    python scripts/plot_results.py [--predictions results/baseline_latex_parse_*.csv]

Outputs:
  docs/figures/parse_baseline_by_type.png   per-equation-type accuracy
  docs/figures/split_composition.png        test-set complexity + equation-type mix
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from math2code.evaluation.eval import load_split  # noqa: E402
from math2code.evaluation.metrics import score_predictions  # noqa: E402

FIG_DIR = Path("docs/figures")
CALC = {"integration", "differential", "derivative", "exponential_decay"}


def load_predictions(path: str) -> dict[str, list]:
    pred: dict[str, list] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pred[row["id"]] = json.loads(row["outputs"])
    return pred


def parse_baseline_figure(predictions: dict[str, list]) -> Path:
    test = load_split("data/split/test.json")
    res = score_predictions(
        [predictions.get(p.task_id, [None] * 5) for p in test],
        [p.test_cases for p in test],
        [p.task_id for p in test],
    )
    per_problem: dict[str, bool] = {}
    for d in res.details:
        per_problem[d.task_id] = per_problem.get(d.task_id, True) and d.passed

    groups: dict[str, tuple[int, int]] = {}
    for p in test:
        g = "calculus" if p.equation_type in CALC else p.equation_type
        ok, n = groups.get(g, (0, 0))
        groups[g] = (ok + (1 if per_problem[p.task_id] else 0), n + 1)

    order = sorted(groups, key=lambda g: -groups[g][1])
    ok = [groups[g][0] for g in order]
    n = [groups[g][1] for g in order]
    acc = [o / cnt * 100 for o, cnt in zip(ok, n)]
    colors = ["#c0392b" if a == 0 else "#2980b9" for a in acc]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    bars = ax.bar(order, acc, color=colors)
    ax.set_ylabel("per-problem accuracy (%)")
    ax.set_title("Zero-cost SymPy parse_latex baseline on the frozen test split")
    ax.set_ylim(0, 105)
    for bar, o, cnt in zip(bars, ok, n):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{o}/{cnt}",
            ha="center",
            fontsize=8,
        )
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out = FIG_DIR / "parse_baseline_by_type.png"
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def split_composition_figure() -> Path:
    FIG_DIR.mkdir(exist_ok=True)
    test = load_split("data/split/test.json")
    cx = Counter(p.complexity for p in test)
    et = Counter(p.equation_type for p in test)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=150)
    a1.bar(
        [f"c{c}" for c in sorted(cx, key=int)],
        [cx[c] for c in sorted(cx, key=int)],
        color="#16a085",
    )
    a1.set_title("Test-set complexity mix")
    a1.set_ylabel("problems")
    labels = sorted(et)
    a2.bar(labels, [et[x] for x in labels], color="#8e44ad")
    a2.set_title("Test-set equation types")
    a2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = FIG_DIR / "split_composition.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="results/baseline_latex_parse_latest.csv")
    ap.add_argument("--composition-only", action="store_true")
    args = ap.parse_args()

    split_composition_figure()
    if not args.composition_only:
        if not Path(args.predictions).exists():
            print(
                f"predictions file not found: {args.predictions} — "
                "run `make baselines` first, then pass --predictions"
            )
            return
        parse_baseline_figure(load_predictions(args.predictions))


if __name__ == "__main__":
    main()

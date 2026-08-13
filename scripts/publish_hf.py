"""Publish the frozen split + verified synthetic pool to the Hugging Face Hub.

Stages a self-contained dataset directory (default: ``hf_staging/``) whose
layout mirrors the published repo:

    README.md                     rendered dataset card (configs: split, synthetic)
    LICENSE                       MIT
    data/
      manifest.json               split SHA-256 manifest (pinned by CI)
      train.jsonl  val.json  test.json  test_ids.txt
      verify_report_combined.json
      verify_sympy_report_fixed.json
      public_test_new_no_sol_no_out.json      (closed-truth probe, no outputs)
      synthetic/*_v1.jsonl                    (26 family pools; mixtures excluded)
      synthetic/*_v1.report.json

Default (no flags) stages + validates everything locally with zero network
traffic; ``--push`` additionally uploads the folder via ``upload_folder``.

Run:
    python scripts/publish_hf.py                                    # stage + validate
    HF_TOKEN=hf_... python scripts/publish_hf.py --push             # stage + validate + upload
    HF_TOKEN=hf_... python scripts/publish_hf.py --push --repo OWNER/NAME

Requires ``huggingface_hub`` (already a dev dependency) and, for push, an
``HF_TOKEN`` with write scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = "Yasirrazaa/math2code-data"
ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "split"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
DATA_DIR = ROOT / "data"
CARD_TEMPLATE = ROOT / "scripts" / "hf_dataset_card_template.md"

# Row count claimed by the dataset card; used as a stale-card guard.
EXPECTED_SYNTHETIC_ROWS = 14_701

LICENSE_TEXT = """MIT License

Copyright (c) 2026 Yasir Raza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Split files published as-is (canonical evidence + data).
SPLIT_FILES = [
    "train.jsonl",
    "val.json",
    "test.json",
    "manifest.json",
    "test_ids.txt",
    "verify_report_combined.json",
    "verify_sympy_report_fixed.json",
]
PROBE_FILE = "public_test_new_no_sol_no_out.json"

# Union schema of the synthetic pool; published rows always carry all keys so
# the 26 family files share one Arrow-compatible schema.
SYNTHETIC_KEYS = [
    "task_id",
    "latex_expression",
    "solution",
    "sympy_exp",
    "truth_code",
    "test_cases",
    "domain",
    "equation_type",
    "complexity",
    "output_type",
    "synthetic",
    "metadata",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def normalize_synthetic_row(row: dict) -> dict:
    """Make a synthetic row schema-uniform and Arrow-loadable.

    The 26 family files share 12 top-level keys but not nested types, and the
    ``datasets`` json builder casts every file to the *first* file's inferred
    schema (structs must match field-for-field). To get one loadable config:

    - ``truth_code``/``sympy_exp`` are ``""`` when absent (string everywhere);
    - ``test_cases[].output`` is the canonical string form of the reference
      metric's ``format_output`` (``parse_number`` round-trips it), so float
      and ``'re+imj'`` rows share one string column;
    - ``test_cases[].input`` and ``metadata`` are lossless JSON strings
      (``json.dumps``), because per-family variable keys (``x`` vs
      ``a,b,m``) and per-family metadata keys cannot share one Arrow struct.
      Solutions declare exact parameter lists, so dummy keys are never added;
      ``json.loads`` restores the dicts losslessly.

    Published rows are normalized; the on-disk source files stay byte-identical.
    """
    from math2code.evaluation.metrics import format_output

    def _canonical(value: object) -> object:
        try:
            return format_output(value)
        except ValueError:
            # Legacy stored strings from before the format fix: 'a+-bj' (no
            # space) is rejected by parse_number; repair to 'a-bj' first.
            s = str(value).replace(" +-", "-").replace("+-", "-")
            return format_output(s)

    out: dict = {}
    for k in SYNTHETIC_KEYS:
        v = row.get(k)
        if v is None and k in ("sympy_exp", "truth_code"):
            v = ""
        elif v is None and k == "metadata":
            v = {}
        out[k] = v
    cases = []
    for tc in out["test_cases"]:
        tc = dict(tc)
        tc["input"] = json.dumps(tc.get("input") or {}, sort_keys=True)
        if tc.get("output") is not None:
            try:
                tc["output"] = _canonical(tc["output"])
            except ValueError as exc:
                raise ValueError(
                    f"unparseable output {tc['output']!r} in task {row.get('task_id')!r}"
                ) from exc
        cases.append(tc)
    out["test_cases"] = cases
    out["metadata"] = json.dumps(out["metadata"], sort_keys=True)
    return out


def stage(staging: Path, repo: str) -> list[tuple[str, int]]:
    """Copy artifacts into the staging dir; return [(rel_path, bytes)]."""
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "data" / "synthetic").mkdir(parents=True)

    copied: list[tuple[str, int]] = []

    def _copy(src: Path, rel: str) -> None:
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append((rel, src.stat().st_size))

    # Frozen split + verification evidence.
    for name in SPLIT_FILES:
        src = SPLIT_DIR / name
        if not src.exists():
            print(f"  [warn] missing split artifact: {src}", file=sys.stderr)
            continue
        _copy(src, f"data/{name}")

    # Closed-truth generalization probe (synthetic rows, no outputs shipped).
    probe = DATA_DIR / PROBE_FILE
    if probe.exists():
        _copy(probe, f"data/{PROBE_FILE}")
    else:
        print(f"  [warn] missing probe: {probe}", file=sys.stderr)

    # Verified synthetic pool: family files only. train_mixture*.jsonl are
    # competition-derived convenience artifacts and are deliberately excluded.
    # Published rows are schema-normalized (see normalize_synthetic_row) but
    # the on-disk source files are left byte-identical.
    family_jsonl = sorted(
        p for p in SYNTHETIC_DIR.glob("*_v1.jsonl") if "train_mixture" not in p.name
    )
    for p in family_jsonl:
        dst = staging / "data" / "synthetic" / p.name
        with p.open() as fh, dst.open("w") as out:
            for line in fh:
                out.write(json.dumps(normalize_synthetic_row(json.loads(line))) + "\n")
        copied.append((f"data/synthetic/{p.name}", dst.stat().st_size))
    for p in sorted(SYNTHETIC_DIR.glob("*_v1.report.json")):
        _copy(p, f"data/synthetic/{p.name}")

    # License + rendered dataset card.
    (staging / "LICENSE").write_text(LICENSE_TEXT)
    copied.append(("LICENSE", len(LICENSE_TEXT.encode())))
    readme = staging / "README.md"
    readme.write_text(render_card(repo))
    copied.append(("README.md", readme.stat().st_size))
    return copied


def render_card(repo: str) -> str:
    tmpl = CARD_TEMPLATE.read_text()
    return tmpl.replace("{{repo}}", repo)


def validate(staging: Path) -> bool:
    """Local integrity pass over the staged tree. Returns True if all good."""
    ok = True
    total_bytes = 0

    def _fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"  [FAIL] {msg}")

    # 1) Split manifest digests must match the committed manifest.
    manifest = json.loads((staging / "data" / "manifest.json").read_text())
    n_bad = 0
    for name, digest in manifest["files"].items():
        p = staging / "data" / name
        if not p.exists():
            _fail(f"manifest entry missing from staging: {name}")
            n_bad += 1
            continue
        if sha256(p) != digest:
            _fail(f"manifest digest mismatch for {name}")
            n_bad += 1
    if n_bad == 0:
        print(f"  [ok] split manifest ({len(manifest['files'])} files) sha256 verified")

    # 2) Every synthetic row parses and carries the full 12-key schema.
    required = set(SYNTHETIC_KEYS)
    total_rows = 0
    row_fail = 0
    for p in sorted((staging / "data" / "synthetic").glob("*_v1.jsonl")):
        n = 0
        for line in p.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row_fail += 1
                continue
            if not required.issubset(row):
                row_fail += 1
            n += 1
        total_rows += n
        print(f"  [ok] {p.name}: {n:>6} rows")
    if row_fail:
        _fail(f"{row_fail} synthetic rows failed parse/schema")
    if total_rows != EXPECTED_SYNTHETIC_ROWS:
        print(
            f"  [warn] synthetic total {total_rows} != card claim "
            f"{EXPECTED_SYNTHETIC_ROWS} — update the dataset card",
            file=sys.stderr,
        )
    else:
        print(f"  [ok] synthetic pool: {total_rows} rows (matches card claim)")

    # 3) Split rows parse; counts match the manifest.
    for name, expected in (
        ("train.jsonl", manifest["n_train"]),
        ("val.json", manifest["n_val"]),
        ("test.json", manifest["n_test"]),
    ):
        p = staging / "data" / name
        if name.endswith(".jsonl"):
            n = sum(1 for _ in p.open())
        else:
            data = json.loads(p.read_text())
            n = len(data)
        status = "ok" if n == expected else f"expected {expected}, got {n}"
        print(f"  [{status.split()[0]:<4}] {name}: {n} rows")
        if n != expected:
            _fail(f"{name} row count mismatch: {status}")

    # 4) Probe parses as a row array and leaks no closed truth.
    probe = staging / "data" / PROBE_FILE
    if probe.exists():
        rows = json.loads(probe.read_text())
        leaked = [
            r.get("task_id")
            for r in rows
            if r.get("solution")
            or r.get("sympy_exp")
            or any(tc.get("output") is not None for tc in (r.get("test_cases") or []))
        ]
        if leaked:
            _fail(
                f"probe contains closed truth ({len(leaked)} rows) — refusing to publish"
            )
        else:
            print(f"  [ok] probe: {len(rows)} rows (closed truth, no outputs)")
    else:
        print("  [warn] probe not staged", file=sys.stderr)

    # 5) Reports parse.
    for p in sorted((staging / "data" / "synthetic").glob("*.report.json")):
        json.loads(p.read_text())
    print(
        f"  [ok] {len(list((staging / 'data' / 'synthetic').glob('*.report.json')))} per-family reports parse"
    )

    for _, size in walk(staging):
        total_bytes += size
    print(f"  [ok] staged total: {total_bytes / 1e6:.1f} MB")
    return ok


def walk(staging: Path) -> list[tuple[Path, int]]:
    return [
        (p, p.stat().st_size)
        for p in sorted(staging.rglob("*"))
        if p.is_file() and p.name != ".gitignore"
    ]


def push(staging: Path, repo: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; export it first (write scope).", file=sys.stderr)
        raise SystemExit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if not api.repo_exists(repo, repo_type="dataset"):
        print(f"creating dataset repo {repo}")
        api.create_repo(repo, repo_type="dataset", private=False)

    files = [p.name for p in sorted(staging.rglob("*")) if p.is_file()]
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(staging),
        commit_message="Publish frozen split + verified synthetic pool (14,701 rows)",
    )
    print(f"uploaded {len(files)} files -> https://huggingface.co/datasets/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        default=os.environ.get("MATH2CODE_HF_REPO", DEFAULT_REPO),
        help="dataset repo id (default: %(default)s)",
    )
    ap.add_argument(
        "--staging",
        default=str(ROOT / "hf_staging"),
        help="staging directory (default: %(default)s)",
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="upload the staged directory to the Hub (requires HF_TOKEN)",
    )
    args = ap.parse_args()

    staging = Path(args.staging)
    print(f"staging into {staging} ...")
    copied = stage(staging, args.repo)
    print(f"  copied {len(copied)} files ({sum(s for _, s in copied) / 1e6:.1f} MB)")
    print("validation:")
    if not validate(staging):
        print("validation FAILED — fix issues before pushing", file=sys.stderr)
        raise SystemExit(2)
    print("validation passed.")

    if args.push:
        push(staging, args.repo)
    else:
        print(
            "\ndry-run complete (no network). Push with: "
            f"HF_TOKEN=hf_... python scripts/publish_hf.py --push --repo {args.repo}"
        )


if __name__ == "__main__":
    main()

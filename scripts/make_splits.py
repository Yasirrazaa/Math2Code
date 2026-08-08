"""Build the frozen train/val/test splits from competition train.json.

- dedup by latex_expression (the file has ~4k duplicates)
- stratified holdout by (domain, complexity): 400 test + 400 val
- writes a manifest with the seed and per-file SHA-256 so the split is frozen
  and auditable

Run:  python scripts/make_splits.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math2code.data.competition import (  # noqa: E402
    DATA_DIR,
    dedup_by_latex,
    load_competition_train,
)

SEED = 42
N_VAL = 400
N_TEST = 400


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rng = random.Random(SEED)
    pairs = dedup_by_latex(load_competition_train())
    print(f"dedup: {len(pairs)} unique-latex samples")

    # stratify by (domain, complexity)
    strata: dict[tuple[str, str], list] = defaultdict(list)
    for p in pairs:
        strata[(p.domain or "unknown", str(p.complexity or "?"))].append(p)

    test: list = []
    val: list = []
    for key, members in sorted(strata.items()):
        rng.shuffle(members)
        quota_test = max(1, int(len(members) * (N_TEST / len(pairs))))
        quota_val = max(1, int(len(members) * (N_VAL / len(pairs))))
        test.extend(members[:quota_test])
        val.extend(members[quota_test : quota_test + quota_val])

    # trim to exact targets
    rng.shuffle(test)
    rng.shuffle(val)
    test = test[:N_TEST]
    val = val[:N_VAL]

    test_ids = {p.task_id for p in test}
    val_ids = {p.task_id for p in val}
    train = [p for p in pairs if p.task_id not in test_ids and p.task_id not in val_ids]
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")

    split_dir = DATA_DIR / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("val.json", val), ("test.json", test)]:
        out = split_dir / name
        with open(out, "w") as f:
            if name.endswith(".jsonl"):
                for r in rows:
                    f.write(json.dumps(r.model_dump()) + "\n")
            else:
                json.dump([r.model_dump() for r in rows], f, indent=2)
        print(f"  wrote {out} ({len(rows)} rows)")

    manifest = {
        "seed": SEED,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "source": "data/train.json (bootcamp competition, deduped by latex)",
        "files": {
            name: sha256(split_dir / name)
            for name in ["train.jsonl", "val.json", "test.json"]
        },
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {split_dir / 'manifest.json'}")
    # freeze: also copy the test set id list for contamination checks
    (split_dir / "test_ids.txt").write_text("\n".join(sorted(test_ids)) + "\n")


if __name__ == "__main__":
    main()

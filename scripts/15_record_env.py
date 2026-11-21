"""Record the exact environment and every hyper-parameter used for the reported runs.

The assignment asks for the model, dependencies, training settings and seed to be
recorded; this writes them into one machine-readable file so a reader does not have
to reconstruct them from prose.

    python scripts/15_record_env.py
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof import __version__  # noqa: E402
from kazprof.lexicon import Lexicon  # noqa: E402

PACKAGES = {
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "torch": "torch",
    "transformers": "transformers",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
}


def versions() -> dict:
    out = {}
    for label, module_name in PACKAGES.items():
        try:
            mod = __import__(module_name)
            out[label] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[label] = "not installed"
    return out


def main() -> int:
    stats = json.loads((ROOT / "data/processed/dataset_stats.json").read_text(encoding="utf-8"))
    train_log = ROOT / "reports/transformer_train.json"
    baseline_log = ROOT / "reports/baseline_val.json"
    final = ROOT / "models/final_model.json"
    lex = Lexicon()

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "kazprof_version": __version__,
        "packages": versions(),
        "random_seeds": {"dataset_split": stats["seed"], "synthetic_generation": stats["seed"],
                         "baseline": 42, "transformer": 42},
        "dataset": {
            "total": stats["total"],
            "labels": stats["labels"],
            "synthetic_rows": stats["synthetic_rows"],
            "splits": {k: v["rows"] for k, v in stats["splits"].items()},
            "near_dup_threshold": stats["near_dup_threshold"],
            "lexicon_terms": len(lex.entries),
            "lexicon_version": lex.version,
        },
        "baseline": json.loads(baseline_log.read_text(encoding="utf-8")) if baseline_log.exists() else None,
        "transformer": json.loads(train_log.read_text(encoding="utf-8")) if train_log.exists() else None,
        "final_model": json.loads(final.read_text(encoding="utf-8")) if final.exists() else None,
    }
    out = ROOT / "reports/training_env.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

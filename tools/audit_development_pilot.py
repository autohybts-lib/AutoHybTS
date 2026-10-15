"""Audit a development-only M4 pilot and its temporal plans without model training."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from autohybts import AutoHybridExperiment
from autohybts.benchmarks.ablation import _configured_cases, _study_config
from autohybts.domain.identity import stable_hash
from autohybts.persistence.manifest import environment, json_safe, source_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = _study_config(json.loads(args.config.read_text(encoding="utf-8")))
    if not config.get("development_only"):
        parser.error("The audit requires development_only=true")
    cases = _configured_cases(config)
    audits, plans = [], []
    for case in cases:
        audits.append(
            {
                "series_id": case.series_id,
                **case.source["data_audit"],
                "original_train_length": case.source["original_train_length"],
                "seasonal_period": case.seasonal_period,
                "index_source": case.source.get("extra", {}).get("index_source"),
            }
        )
        settings = {
            **config["space"],
            "fh": case.horizon,
            "budget": config["budget"],
            "session_id": config["seeds"][0],
            "seasonal_period": case.seasonal_period,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exp = AutoHybridExperiment(**settings, test_size=case.test_size).fit(case.values)
            exp.splits.reset_index().to_csv(
                args.outdir / f"{case.series_id}-splits.csv", index=False
            )
            inner = AutoHybridExperiment(**settings, test_size=0).fit(
                case.values.iloc[: -case.test_size - config["outer_folds"] * case.horizon]
            )
        plans.append(
            {
                "series_id": case.series_id,
                "horizon": case.horizon,
                "folds_per_layer": exp.fold_,
                "earliest_inner_folds": inner.fold_,
                "warnings": [str(w.message) for w in caught],
                "official_test_accessed": False,
                "training_performed": False,
            }
        )
    audit_table = pd.DataFrame(audits)
    audit_table.to_csv(args.outdir / "data_audit.csv", index=False)
    report = {
        "status": "READY_WITH_WARNINGS"
        if any(p["warnings"] for p in plans) or any(a["status"] != "READY" for a in audits)
        else "READY",
        "scope": "development_cost_and_workflow_pilot",
        "resolved_config": config,
        "config_hash": stable_hash(config),
        "environment": environment(),
        "source_fingerprint": source_fingerprint(),
        "temporal_plans": plans,
        "data_audit": audits,
        "official_test_accessed": False,
        "sources": {c.series_id: c.source for c in cases},
    }
    (args.outdir / "preflight.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if args.plot:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(7, 3))
        ax = fig.subplots()
        ax.bar(audit_table["series_id"], audit_table["observations"])
        ax.set_ylabel("Development observations")
        ax.set_title("Pilot input lengths (official test excluded)")
        fig.savefig(args.outdir / "data_audit.png", bbox_inches="tight", dpi=120)
    print(json.dumps({"status": report["status"], "plans": plans}, ensure_ascii=False))


if __name__ == "__main__":
    main()

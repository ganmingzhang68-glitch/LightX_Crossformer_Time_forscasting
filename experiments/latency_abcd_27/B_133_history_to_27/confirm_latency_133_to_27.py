from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from latency_fusion_common import ExperimentRepository
from latency_133_to_27_common import TargetContract, prepare_fold_data_27, train_fold_27


FOLDS = ("fold_a", "fold_b", "fold_c")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm the selected 133-to-27 HPO parameters on larger fold samples."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--best-trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-train-windows-per-device", type=int, default=12000)
    parser.add_argument("--max-validation-windows-per-device", type=int, default=6000)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience) <= 0:
        parser.error("epochs and patience must be positive")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("133-to-27 confirmation requires CUDA")

    best_path = args.best_trial.resolve()
    best = json.loads(best_path.read_text(encoding="utf-8"))
    params = dict(best["params"])
    repository = ExperimentRepository(args.data_root.resolve(), include_external=False)
    contract = TargetContract(args.targets.resolve(), repository)
    if best.get("data_manifest_sha256") != repository.manifest["manifest_sha256"]:
        raise ValueError("HPO and confirmation data manifests do not match")
    if best.get("target_features") != list(contract.names):
        raise ValueError("HPO and confirmation target contracts do not match")

    fold_data = {
        name: prepare_fold_data_27(
            repository,
            contract,
            name,
            args.max_train_windows_per_device,
            args.max_validation_windows_per_device,
        )
        for name in FOLDS
    }
    output.mkdir(parents=True)
    results = []
    for index, name in enumerate(FOLDS):
        results.append(
            train_fold_27(
                repository,
                contract,
                fold_data[name],
                args.library_root.resolve(),
                params,
                output / name,
                args.epochs,
                min(args.patience, args.epochs),
                args.num_workers,
                args.seed + index,
                not args.no_amp,
            )
        )

    objectives = np.asarray(
        [item["best_metrics"]["objective_selected_27"] for item in results],
        dtype=np.float64,
    )
    report = {
        "source_best_trial": str(best_path),
        "source_hpo_trial": int(best["best_trial"]),
        "params": params,
        "aggregate_objective": 0.75 * float(objectives.mean()) + 0.25 * float(objectives.max()),
        "mean_fold_objective": float(objectives.mean()),
        "worst_fold_objective": float(objectives.max()),
        "fold_results": results,
        "max_train_windows_per_device": args.max_train_windows_per_device,
        "max_validation_windows_per_device": args.max_validation_windows_per_device,
        "epochs": args.epochs,
        "patience": args.patience,
        "input_channels": 133,
        "output_channels": 27,
        "target_features": list(contract.names),
        "target_input_indices": contract.input_indices.tolist(),
        "device04_loaded_or_used": False,
        "device04_is_clean_external_test": False,
        "data_manifest_sha256": repository.manifest["manifest_sha256"],
    }
    (output / "confirmation_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

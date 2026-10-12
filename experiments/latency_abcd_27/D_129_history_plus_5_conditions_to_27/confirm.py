from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiment_common import FOLDS, ExperimentContract, ExperimentRepository, prepare_fold_data, train_fold


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Confirm selected HPO parameters on larger fold samples.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
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
    if not torch.cuda.is_available():
        raise RuntimeError("Confirmation requires CUDA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    best_path = args.best_trial.resolve()
    best = json.loads(best_path.read_text(encoding="utf-8"))
    contract = ExperimentContract(here / "experiment_spec.json", args.data_root)
    repository = ExperimentRepository(contract, include_external=False)
    if best.get("data_manifest_sha256") != repository.manifest["manifest_sha256"]:
        raise ValueError("HPO and confirmation data manifests do not match")
    if best.get("contract") != contract.describe():
        raise ValueError("HPO and confirmation experiment contracts do not match")
    params = dict(best["params"])
    fold_data = {name: prepare_fold_data(repository, name,
                 args.max_train_windows_per_device, args.max_validation_windows_per_device)
                 for name in FOLDS}
    output.mkdir(parents=True)
    results = [train_fold(repository, fold_data[name], args.library_root.resolve(), params,
               output / name, args.epochs, min(args.patience, args.epochs), args.num_workers,
               args.seed + index, not args.no_amp) for index, name in enumerate(FOLDS)]
    objectives = np.asarray([r["best_metrics"]["objective_selected_27"] for r in results])
    report = {"source_best_trial": str(best_path), "source_hpo_trial": int(best["best_trial"]),
              "params": params,
              "aggregate_objective": 0.75 * float(objectives.mean()) + 0.25 * float(objectives.max()),
              "mean_fold_objective": float(objectives.mean()),
              "worst_fold_objective": float(objectives.max()), "fold_results": results,
              "max_train_windows_per_device": args.max_train_windows_per_device,
              "max_validation_windows_per_device": args.max_validation_windows_per_device,
              "epochs": args.epochs, "patience": args.patience, "contract": contract.describe(),
              "device04_loaded_or_used": False, "device04_is_clean_external_test": False,
              "data_manifest_sha256": repository.manifest["manifest_sha256"]}
    (output / "confirmation_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

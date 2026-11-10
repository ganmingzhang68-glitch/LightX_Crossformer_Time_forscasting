from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from latency_fusion_common import ExperimentRepository
from latency_133_to_27_common import (
    TargetContract,
    prepare_fold_data_27,
    train_fold_27,
)


FOLDS = ("fold_a", "fold_b", "fold_c")


def sample_params(trial) -> dict:
    d_model = trial.suggest_categorical("d_model", [64, 96, 128])
    heads = [value for value in (2, 4, 8) if d_model % value == 0]
    multiplier = trial.suggest_categorical("d_ff_multiplier", [2, 4])
    continuous_weight = trial.suggest_categorical(
        "continuous_loss_weight", [0.6, 0.7, 0.8, 0.9]
    )
    return {
        "d_model": d_model,
        "n_heads": trial.suggest_categorical("n_heads", heads),
        "e_layers": trial.suggest_int("e_layers", 1, 3),
        "d_ff": d_model * multiplier,
        "d_ff_multiplier": multiplier,
        "factor": trial.suggest_categorical("factor", [1, 3, 5]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 3e-4, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "huber_delta": trial.suggest_categorical("huber_delta", [0.5, 1.0, 2.0]),
        "continuous_loss_weight": continuous_weight,
        "boolean_loss_weight": 1 - continuous_weight,
        "crossformer_seg_len": trial.suggest_categorical("crossformer_seg_len", [6, 10, 15]),
        "revin_affine": trial.suggest_categorical("revin_affine", [False, True]),
        "use_revin": True,
        "residual_forecast": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-train-windows-per-device", type=int, default=3000)
    parser.add_argument("--max-validation-windows-per-device", type=int, default=1500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("133-to-27 HPO requires CUDA")
    output = args.output.resolve()
    repository = ExperimentRepository(args.data_root.resolve(), include_external=False)
    contract = TargetContract(args.targets, repository)
    fold_data = {
        name: prepare_fold_data_27(
            repository, contract, name,
            args.max_train_windows_per_device,
            args.max_validation_windows_per_device,
        ) for name in FOLDS
    }
    print(json.dumps({
        "input_channels": 133,
        "output_channels": 27,
        "targets": list(contract.names),
        "target_input_indices": contract.input_indices.tolist(),
        "device04_loaded_or_used": False,
        "folds": {name: {
            "train_devices": list(data.train_devices),
            "validation_devices": list(data.validation_devices),
            "train_windows": len(data.train_dataset),
            "validation_windows": len(data.validation_dataset),
        } for name, data in fold_data.items()},
    }, ensure_ascii=False, indent=2), flush=True)

    seed_params = {
        "d_model": 96, "n_heads": 4, "e_layers": 2, "d_ff": 192,
        "d_ff_multiplier": 2, "factor": 1, "dropout": 0.05,
        "learning_rate": 8.75107062101344e-05, "weight_decay": 0.0,
        "batch_size": 16, "huber_delta": 2.0,
        "continuous_loss_weight": 0.8, "boolean_loss_weight": 0.2,
        "crossformer_seg_len": 10, "revin_affine": False,
        "use_revin": True, "residual_forecast": True,
    }
    if args.check_only:
        if output.exists():
            raise FileExistsError(output)
        result = train_fold_27(
            repository, contract, fold_data["fold_a"], args.library_root,
            seed_params, output, 1, 1, 0, args.seed, not args.no_amp
        )
        print(json.dumps({"check_only": True, "result": result}, ensure_ascii=False, indent=2))
        return

    import optuna
    output.mkdir(parents=True, exist_ok=args.resume)
    trials_root = output / "trials"
    trials_root.mkdir(exist_ok=True)
    storage = output / "study.sqlite3"
    if storage.exists() and not args.resume:
        raise FileExistsError(storage)
    study = optuna.create_study(
        study_name="crossformer_latency_133_to_selected_27",
        storage=f"sqlite:///{storage.as_posix()}", direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
        load_if_exists=args.resume,
    )
    if not args.resume and len(study.trials) == 0:
        study.enqueue_trial({k: v for k, v in seed_params.items() if k not in (
            "d_ff", "boolean_loss_weight", "use_revin", "residual_forecast"
        )})

    def objective(trial):
        params = sample_params(trial)
        trial_dir = trials_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir()
        (trial_dir / "params.json").write_text(
            json.dumps({"trial": trial.number, "params": params}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        results = []
        try:
            for index, fold in enumerate(FOLDS):
                result = train_fold_27(
                    repository, contract, fold_data[fold], args.library_root,
                    params, trial_dir / fold, args.epochs,
                    min(args.patience, args.epochs), args.num_workers,
                    args.seed + trial.number * 100 + index, not args.no_amp,
                )
                results.append(result)
                partial = float(np.mean([
                    item["best_metrics"]["objective_selected_27"] for item in results
                ]))
                trial.report(partial, step=index)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned after {fold}")
            values = np.asarray([
                item["best_metrics"]["objective_selected_27"] for item in results
            ])
            aggregate = 0.75 * float(values.mean()) + 0.25 * float(values.max())
            document = {
                "trial": trial.number, "aggregate_objective": aggregate,
                "mean_fold_objective": float(values.mean()),
                "worst_fold_objective": float(values.max()),
                "fold_results": results, "device04_loaded_or_used": False,
            }
            (trial_dir / "result.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"trial={trial.number} aggregate={aggregate:.6f}", flush=True)
            return aggregate
        finally:
            torch.cuda.empty_cache()

    study.optimize(objective, n_trials=args.trials, n_jobs=1)
    study.trials_dataframe().to_csv(output / "trials.csv", index=False, encoding="utf-8-sig")
    best_dir = trials_root / f"trial_{study.best_trial.number:04d}"
    params_doc = json.loads((best_dir / "params.json").read_text(encoding="utf-8"))
    result_doc = json.loads((best_dir / "result.json").read_text(encoding="utf-8"))
    best = {
        "best_trial": study.best_trial.number,
        "aggregate_objective": study.best_value,
        "params": params_doc["params"],
        "fold_result": result_doc,
        "input_channels": 133,
        "output_channels": 27,
        "target_features": list(contract.names),
        "target_input_indices": contract.input_indices.tolist(),
        "device04_loaded_or_used": False,
        "device04_is_clean_external_test": False,
        "data_manifest_sha256": repository.manifest["manifest_sha256"],
    }
    (output / "best_trial.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"133-to-27 HPO failed: {exc}", file=sys.stderr)
        raise

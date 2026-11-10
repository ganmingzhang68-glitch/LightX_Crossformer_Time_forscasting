from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiment_common import FOLDS, ExperimentContract, ExperimentRepository, prepare_fold_data, train_fold


def sample_params(trial, has_conditions: bool) -> dict:
    d_model = trial.suggest_categorical("d_model", [64, 96, 128])
    multiplier = trial.suggest_categorical("d_ff_multiplier", [2, 4])
    continuous_weight = trial.suggest_categorical("continuous_loss_weight", [0.6, 0.7, 0.8, 0.9])
    params = {
        "d_model": d_model,
        "n_heads": trial.suggest_categorical("n_heads", [h for h in (2, 4, 8) if d_model % h == 0]),
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
        "boolean_loss_weight": 1.0 - continuous_weight,
        "crossformer_seg_len": trial.suggest_categorical("crossformer_seg_len", [6, 10, 15]),
        "revin_affine": trial.suggest_categorical("revin_affine", [False, True]),
        "use_revin": True,
        "residual_forecast": True,
    }
    if has_conditions:
        params.update({
            "condition_hidden": trial.suggest_categorical("condition_hidden", [16, 32, 64]),
            "condition_layers": trial.suggest_int("condition_layers", 1, 2),
            "condition_dropout": trial.suggest_categorical("condition_dropout", [0.0, 0.1]),
            "condition_embedding_dim": trial.suggest_categorical("condition_embedding_dim", [2, 4, 8]),
            "condition_gate_init": 0.1,
        })
    return params


def seed_params(has_conditions: bool) -> dict:
    params = {
        "d_model": 128, "n_heads": 2, "e_layers": 2, "d_ff": 256,
        "d_ff_multiplier": 2, "factor": 5, "dropout": 0.0,
        "learning_rate": 2.9068076540795856e-4, "weight_decay": 0.0,
        "batch_size": 32, "huber_delta": 2.0,
        "continuous_loss_weight": 0.9, "boolean_loss_weight": 0.1,
        "crossformer_seg_len": 15, "revin_affine": True,
        "use_revin": True, "residual_forecast": True,
    }
    if has_conditions:
        params.update({"condition_hidden": 32, "condition_layers": 1,
                       "condition_dropout": 0.0, "condition_embedding_dim": 4,
                       "condition_gate_init": 0.1})
    return params


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Tune one self-contained A/B/C/D experiment.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-train-windows-per-device", type=int, default=3000)
    parser.add_argument("--max-validation-windows-per-device", type=int, default=1500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if min(args.trials, args.epochs, args.patience) <= 0:
        parser.error("trials, epochs and patience must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("HPO requires CUDA")

    contract = ExperimentContract(here / "experiment_spec.json", args.data_root)
    repository = ExperimentRepository(contract, include_external=False)
    fold_data = {
        name: prepare_fold_data(repository, name, args.max_train_windows_per_device,
                                args.max_validation_windows_per_device)
        for name in FOLDS
    }
    print(json.dumps({"contract": contract.describe(), "device04_loaded_or_used": False,
                      "folds": {name: {"train_devices": list(data.train_devices),
                      "validation_devices": list(data.validation_devices),
                      "train_windows": len(data.train_dataset),
                      "validation_windows": len(data.validation_dataset)}
                      for name, data in fold_data.items()}}, ensure_ascii=False, indent=2), flush=True)

    import optuna
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    trials_root = output / "trials"
    trials_root.mkdir(exist_ok=True)
    storage = output / "study.sqlite3"
    if storage.exists() and not args.resume:
        raise FileExistsError(storage)
    study = optuna.create_study(
        study_name=f"crossformer_{contract.spec['experiment_id']}",
        storage=f"sqlite:///{storage.as_posix()}", direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
        load_if_exists=args.resume,
    )
    seeded = seed_params(bool(contract.condition_features))
    derived = {"d_ff", "boolean_loss_weight", "use_revin", "residual_forecast", "condition_gate_init"}
    if not args.resume and len(study.trials) == 0:
        study.enqueue_trial({k: v for k, v in seeded.items() if k not in derived})

    def objective(trial):
        params = sample_params(trial, bool(contract.condition_features))
        trial_dir = trials_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir()
        (trial_dir / "params.json").write_text(json.dumps({"trial": trial.number, "params": params},
            ensure_ascii=False, indent=2), encoding="utf-8")
        results = []
        try:
            for index, fold_name in enumerate(FOLDS):
                result = train_fold(repository, fold_data[fold_name], args.library_root.resolve(),
                    params, trial_dir / fold_name, args.epochs, min(args.patience, args.epochs),
                    args.num_workers, args.seed + trial.number * 100 + index, not args.no_amp)
                results.append(result)
                partial = float(np.mean([r["best_metrics"]["objective_selected_27"] for r in results]))
                trial.report(partial, step=index)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned after {fold_name}")
            values = np.asarray([r["best_metrics"]["objective_selected_27"] for r in results])
            aggregate = 0.75 * float(values.mean()) + 0.25 * float(values.max())
            document = {"trial": trial.number, "aggregate_objective": aggregate,
                        "mean_fold_objective": float(values.mean()),
                        "worst_fold_objective": float(values.max()), "fold_results": results,
                        "device04_loaded_or_used": False}
            (trial_dir / "result.json").write_text(json.dumps(document, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
            return aggregate
        finally:
            torch.cuda.empty_cache()

    study.optimize(objective, n_trials=args.trials, n_jobs=1)
    study.trials_dataframe().to_csv(output / "trials.csv", index=False, encoding="utf-8-sig")
    best_dir = trials_root / f"trial_{study.best_trial.number:04d}"
    params_doc = json.loads((best_dir / "params.json").read_text(encoding="utf-8"))
    result_doc = json.loads((best_dir / "result.json").read_text(encoding="utf-8"))
    best = {"best_trial": study.best_trial.number, "aggregate_objective": study.best_value,
            "params": params_doc["params"], "fold_result": result_doc,
            "contract": contract.describe(), "device04_loaded_or_used": False,
            "device04_is_clean_external_test": False,
            "data_manifest_sha256": repository.manifest["manifest_sha256"]}
    (output / "best_trial.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"HPO failed: {exc}", file=sys.stderr)
        raise

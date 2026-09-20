from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from experiment_common import (ExperimentContract, ExperimentModel, ExperimentRepository,
    ExperimentWindowDataset, GroupedHuberLoss, evaluate, make_loader, set_seed)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train one final A/B/C/D model after HPO confirmation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--best-trial", type=Path, required=True)
    parser.add_argument("--confirmation-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-train-windows-per-device", type=int, default=0)
    parser.add_argument("--final-epochs", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--evaluate-device04", action="store_true",
                        help="Exploratory only: device 04 helped select the 27 targets.")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("Final training requires CUDA")

    best_path = args.best_trial.resolve()
    best = json.loads(best_path.read_text(encoding="utf-8"))
    params = dict(best["params"])
    fold_results = best["fold_result"]["fold_results"]
    epoch_source = str(best_path)
    if args.confirmation_result:
        confirmation_path = args.confirmation_result.resolve()
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        if confirmation.get("device04_loaded_or_used") is not False:
            raise ValueError("Confirmation must not use device 04")
        if confirmation.get("params") != params:
            raise ValueError("Confirmation parameters do not match HPO")
        if confirmation.get("data_manifest_sha256") != best.get("data_manifest_sha256"):
            raise ValueError("Confirmation and HPO manifests do not match")
        fold_results = confirmation["fold_results"]
        epoch_source = str(confirmation_path)
    fold_epochs = [int(r["best_epoch"]) for r in fold_results]
    final_epochs = args.final_epochs if args.final_epochs > 0 else max(1, int(statistics.median(fold_epochs)))

    contract = ExperimentContract(here / "experiment_spec.json", args.data_root)
    repository = ExperimentRepository(contract, include_external=args.evaluate_device04)
    if best.get("data_manifest_sha256") != repository.manifest["manifest_sha256"]:
        raise ValueError("HPO and final-training manifests do not match")
    if best.get("contract") != contract.describe():
        raise ValueError("HPO and final-training contracts do not match")
    mean, scale = repository.load_scaler("final")
    train_devices = tuple(repository.manifest["training_devices"])
    scaled = {name: ((repository.raw_values[name] - mean) / scale).astype(np.float32)
              for name in train_devices}
    dataset = ExperimentWindowDataset(scaled, repository.windows, train_devices, contract,
        args.max_train_windows_per_device or None, True)
    set_seed(args.seed)
    device = torch.device("cuda")
    model = ExperimentModel(contract, args.library_root.resolve(), params, mean, scale).to(device)
    criterion = GroupedHuberLoss(params["huber_delta"], params["continuous_loss_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs)
    use_amp = not args.no_amp
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = make_loader(dataset, params["batch_size"], True, args.num_workers, args.seed)
    output.mkdir(parents=True)
    history = []
    for epoch in range(1, final_epochs + 1):
        model.train(); sums = np.zeros(3); batches = 0
        for dynamic, condition, target in loader:
            dynamic = dynamic.float().to(device, non_blocking=True)
            condition = condition.float().to(device, non_blocking=True)
            target = target.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                components = criterion.forward_with_components(model(dynamic, condition), target)
            amp_scaler.scale(components[0]).backward(); amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer); amp_scaler.update()
            sums += [float(item.detach()) for item in components]; batches += 1
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
               "total_loss": float(sums[0] / batches),
               "continuous_loss": float(sums[1] / batches),
               "boolean_loss": float(sums[2] / batches)}
        history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

    torch.save(model.state_dict(), output / "model.pth")
    np.savez_compressed(output / "scaler.npz", mean=mean, scale=scale)
    config = {"params": params, "fold_best_epochs": fold_epochs, "final_epochs": final_epochs,
              "epoch_selection_source": epoch_source, "seed": args.seed,
              "train_devices": list(train_devices), "train_windows": len(dataset),
              "train_windows_by_device": dataset.selected_counts, "contract": contract.describe(),
              "manifest_sha256": repository.manifest["manifest_sha256"],
              "device04_used_during_hpo_or_confirmation": False,
              "device04_is_clean_external_test": False}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.evaluate_device04:
        external = repository.manifest["external_test_device"]
        external_scaled = {external: ((repository.raw_values[external] - mean) / scale).astype(np.float32)}
        external_dataset = ExperimentWindowDataset(external_scaled, repository.windows,
            (external,), contract, None, False)
        metrics = evaluate(model, make_loader(external_dataset, params["batch_size"], False,
            args.num_workers, args.seed), contract, mean, scale, device, use_amp)
        report = {"device": external, "exploratory_evaluation": True, "clean_external_test": False,
                  "reason": "Device 04 was used by the historical target-selection procedure.",
                  "windows": len(external_dataset), "metrics": metrics}
        (output / "device04_exploratory_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

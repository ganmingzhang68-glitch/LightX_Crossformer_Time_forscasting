from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from latency_fusion_common import ExperimentRepository, make_loader, set_seed
from latency_133_to_27_common import (
    GroupedTargetHuberLoss,
    SelectedOutputCrossformer,
    SelectedTargetWindowDataset,
    TargetContract,
    evaluate_model_27,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the final Crossformer with 133 inputs and 27 selected outputs."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--best-trial", type=Path, required=True)
    parser.add_argument("--confirmation-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-train-windows-per-device", type=int, default=0)
    parser.add_argument("--final-epochs", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--evaluate-device04",
        action="store_true",
        help="Exploratory evaluation only: device 04 helped select these 27 targets.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    best_path = args.best_trial.resolve()
    best = json.loads(best_path.read_text(encoding="utf-8"))
    params = dict(best["params"])

    epoch_source = str(best_path)
    if args.confirmation_result:
        confirmation_path = args.confirmation_result.resolve()
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        if confirmation.get("device04_loaded_or_used") is not False:
            raise ValueError("Confirmation must not use device 04")
        if confirmation.get("params") != params:
            raise ValueError("Confirmation parameters do not match the selected HPO trial")
        if confirmation.get("data_manifest_sha256") != best.get("data_manifest_sha256"):
            raise ValueError("Confirmation and HPO data manifests do not match")
        fold_results = confirmation["fold_results"]
        epoch_source = str(confirmation_path)
    else:
        fold_results = best["fold_result"]["fold_results"]
    fold_epochs = [int(item["best_epoch"]) for item in fold_results]
    final_epochs = args.final_epochs if args.final_epochs > 0 else max(
        1, int(statistics.median(fold_epochs))
    )

    repository = ExperimentRepository(
        args.data_root.resolve(), include_external=args.evaluate_device04
    )
    contract = TargetContract(args.targets.resolve(), repository)
    if best.get("data_manifest_sha256") != repository.manifest["manifest_sha256"]:
        raise ValueError("HPO and final-training data manifests do not match")
    if best.get("target_features") != list(contract.names):
        raise ValueError("HPO and final-training target contracts do not match")

    mean, scale = repository.load_scaler("final")
    train_devices = tuple(repository.manifest["training_devices"])
    scaled = {
        name: ((repository.raw_values[name] - mean) / scale).astype(np.float32)
        for name in train_devices
    }
    train_dataset = SelectedTargetWindowDataset(
        scaled,
        repository.windows,
        train_devices,
        contract.input_indices,
        args.max_train_windows_per_device,
        True,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Final 133-to-27 training requires CUDA")
    model = SelectedOutputCrossformer(
        args.library_root.resolve(), params, contract.input_indices
    ).to(device)
    criterion = GroupedTargetHuberLoss(
        float(params["huber_delta"]), float(params["continuous_loss_weight"])
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs)
    use_amp = not args.no_amp
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = make_loader(
        train_dataset, int(params["batch_size"]), True, args.num_workers, args.seed
    )
    output.mkdir(parents=True)
    history = []
    for epoch in range(1, final_epochs + 1):
        model.train()
        sums = np.zeros(3, dtype=np.float64)
        batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                components = criterion.forward_with_components(
                    model(batch_x, None, None, None), batch_y
                )
            amp_scaler.scale(components[0]).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            sums += [float(item.detach().item()) for item in components]
            batches += 1
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "total_loss": float(sums[0] / batches),
            "continuous_loss": float(sums[1] / batches),
            "boolean_loss": float(sums[2] / batches),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    torch.save(model.state_dict(), output / "model.pth")
    np.savez_compressed(output / "scaler.npz", mean=mean, scale=scale)
    config = {
        "params": params,
        "fold_best_epochs": fold_epochs,
        "final_epochs": final_epochs,
        "epoch_selection_source": epoch_source,
        "seed": args.seed,
        "train_devices": list(train_devices),
        "train_windows": len(train_dataset),
        "train_windows_by_device": train_dataset.selected_counts,
        "input_channels": 133,
        "input_feature_order": repository.feature_names,
        "output_channels": 27,
        "output_feature_order": list(contract.names),
        "target_input_indices": contract.input_indices.tolist(),
        "manifest_sha256": repository.manifest["manifest_sha256"],
        "device04_used_during_hpo_or_confirmation": False,
        "device04_is_clean_external_test": False,
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.evaluate_device04:
        external = repository.manifest["external_test_device"]
        external_scaled = {
            external: ((repository.raw_values[external] - mean) / scale).astype(np.float32)
        }
        external_dataset = SelectedTargetWindowDataset(
            external_scaled,
            repository.windows,
            (external,),
            contract.input_indices,
            None,
            False,
        )
        metrics = evaluate_model_27(
            model,
            make_loader(
                external_dataset,
                int(params["batch_size"]),
                False,
                args.num_workers,
                args.seed,
            ),
            contract,
            mean,
            scale,
            device,
            use_amp,
        )
        report = {
            "device": external,
            "exploratory_evaluation": True,
            "clean_external_test": False,
            "reason": "Device 04 was used by the historical target-selection procedure.",
            "windows": len(external_dataset),
            "metrics": metrics,
        }
        (output / "device04_exploratory_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from latency_fusion_common import (
    ExperimentRepository,
    build_model,
    evenly_spaced_indices,
    make_loader,
    set_seed,
)


class TargetContract:
    def __init__(self, path: Path, repository: ExperimentRepository):
        document = json.loads(path.resolve().read_text(encoding="utf-8"))
        self.continuous_names = tuple(document["continuous_features"])
        self.boolean_names = tuple(document["boolean_features"])
        self.names = (*self.continuous_names, *self.boolean_names)
        if len(self.continuous_names) != 22 or len(self.boolean_names) != 5:
            raise ValueError("Expected exactly 22 continuous and 5 boolean targets")
        if len(set(self.names)) != 27:
            raise ValueError("The target contract contains duplicate fields")
        missing = [name for name in self.names if name not in repository.feature_names]
        if missing:
            raise ValueError(f"Targets missing from the 133-channel manifest: {missing}")
        self.input_indices = np.asarray(
            [repository.feature_names.index(name) for name in self.names], dtype=np.int64
        )
        if np.any(self.input_indices >= 128):
            raise ValueError("The historical 27-target contract must be contained in old128")
        self.continuous_indices = np.arange(22, dtype=np.int64)
        self.boolean_indices = np.arange(22, 27, dtype=np.int64)
        self.document = document


class SelectedTargetWindowDataset(Dataset):
    def __init__(
        self,
        scaled_values: dict[str, np.ndarray],
        windows: dict,
        devices: tuple[str, ...],
        target_input_indices: np.ndarray,
        max_windows_per_device: int | None,
        balance_devices: bool,
    ):
        counts = {device: len(windows[device]) for device in devices}
        if not devices or min(counts.values()) <= 0:
            raise ValueError(f"Invalid devices or empty windows: {counts}")
        target = min(counts.values()) if balance_devices and len(devices) > 1 else None
        if max_windows_per_device and max_windows_per_device > 0:
            target = min(target, max_windows_per_device) if target else max_windows_per_device
        arrays, codes, past, future_start, future_end = [], [], [], [], []
        selected_counts = {}
        for code, device in enumerate(devices):
            frame = windows[device]
            positions = evenly_spaced_indices(len(frame), target)
            selected = frame.iloc[positions]
            arrays.append(scaled_values[device])
            codes.append(np.full(len(selected), code, dtype=np.int16))
            past.append(selected.past_start_row.to_numpy(dtype=np.int64))
            future_start.append(selected.future_start_row.to_numpy(dtype=np.int64))
            future_end.append(selected.future_end_row_exclusive.to_numpy(dtype=np.int64))
            selected_counts[device] = len(selected)
        self.data_arrays = arrays
        self.devices = devices
        self.device_codes = np.concatenate(codes)
        self.past_starts = np.concatenate(past)
        self.future_starts = np.concatenate(future_start)
        self.future_ends = np.concatenate(future_end)
        self.target_input_indices = target_input_indices
        self.selected_counts = selected_counts

    def __len__(self) -> int:
        return len(self.device_codes)

    def __getitem__(self, index: int):
        code = int(self.device_codes[index])
        values = self.data_arrays[code]
        past_start = int(self.past_starts[index])
        future_start = int(self.future_starts[index])
        future_end = int(self.future_ends[index])
        x = values[past_start : past_start + 150]
        y = values[future_start:future_end, self.target_input_indices]
        if x.shape != (150, 133) or y.shape != (60, 27):
            raise RuntimeError(f"Unexpected window shapes: x={x.shape}, y={y.shape}")
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class FoldData27:
    name: str
    train_devices: tuple[str, ...]
    validation_devices: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    scaled_values: dict[str, np.ndarray]
    train_dataset: SelectedTargetWindowDataset
    validation_dataset: SelectedTargetWindowDataset


def prepare_fold_data_27(
    repository: ExperimentRepository,
    contract: TargetContract,
    fold_name: str,
    max_train_windows_per_device: int | None,
    max_validation_windows_per_device: int | None,
) -> FoldData27:
    fold = repository.manifest["folds"][fold_name]
    train_devices = tuple(fold["train_devices"])
    validation_devices = tuple(fold["validation_devices"])
    mean, scale = repository.load_scaler(fold_name)
    required = (*train_devices, *validation_devices)
    scaled = {
        device: ((repository.raw_values[device] - mean) / scale).astype(np.float32)
        for device in required
    }
    return FoldData27(
        name=fold_name,
        train_devices=train_devices,
        validation_devices=validation_devices,
        mean=mean,
        scale=scale,
        scaled_values=scaled,
        train_dataset=SelectedTargetWindowDataset(
            scaled,
            repository.windows,
            train_devices,
            contract.input_indices,
            max_train_windows_per_device,
            True,
        ),
        validation_dataset=SelectedTargetWindowDataset(
            scaled,
            repository.windows,
            validation_devices,
            contract.input_indices,
            max_validation_windows_per_device,
            False,
        ),
    )


class SelectedOutputCrossformer(nn.Module):
    def __init__(self, library_root: Path, params: dict, target_input_indices: np.ndarray):
        super().__init__()
        self.backbone = build_model(library_root, params)
        self.register_buffer(
            "target_input_indices",
            torch.as_tensor(target_input_indices, dtype=torch.long),
            persistent=True,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        full = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return torch.index_select(full, dim=-1, index=self.target_input_indices)


class GroupedTargetHuberLoss(nn.Module):
    def __init__(self, delta: float, continuous_weight: float):
        super().__init__()
        if delta <= 0 or not 0 < continuous_weight < 1:
            raise ValueError("Invalid Huber loss settings")
        self.delta = float(delta)
        self.continuous_weight = float(continuous_weight)

    def forward_with_components(self, prediction: torch.Tensor, target: torch.Tensor):
        elementwise = F.huber_loss(prediction, target, reduction="none", delta=self.delta)
        continuous = elementwise[..., :22].mean()
        boolean = elementwise[..., 22:].mean()
        total = self.continuous_weight * continuous + (1 - self.continuous_weight) * boolean
        return total, continuous, boolean

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return self.forward_with_components(prediction, target)[0]


def evaluate_model_27(
    model: nn.Module,
    loader,
    contract: TargetContract,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> dict:
    count = 27
    squared = np.zeros(count, dtype=np.float64)
    baseline_squared = np.zeros(count, dtype=np.float64)
    absolute = np.zeros(count, dtype=np.float64)
    baseline_absolute = np.zeros(count, dtype=np.float64)
    tp = np.zeros(5, dtype=np.int64)
    fp = np.zeros(5, dtype=np.int64)
    fn = np.zeros(5, dtype=np.int64)
    baseline_tp = np.zeros(5, dtype=np.int64)
    baseline_fp = np.zeros(5, dtype=np.int64)
    baseline_fn = np.zeros(5, dtype=np.int64)
    positives = np.zeros(5, dtype=np.int64)
    point_count = 0
    thresholds = (
        0.5 - scaler_mean[contract.input_indices[contract.boolean_indices]]
    ) / scaler_scale[contract.input_indices[contract.boolean_indices]]
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(batch_x, None, None, None)
            prediction = output.float().cpu().numpy().reshape(-1, count)
            target = batch_y.float().cpu().numpy().reshape(-1, count)
            baseline = (
                batch_x[:, -1:, contract.input_indices]
                .expand(-1, batch_y.shape[1], -1)
                .float().cpu().numpy().reshape(-1, count)
            )
            error = prediction.astype(np.float64) - target.astype(np.float64)
            baseline_error = baseline.astype(np.float64) - target.astype(np.float64)
            squared += np.square(error).sum(axis=0)
            baseline_squared += np.square(baseline_error).sum(axis=0)
            absolute += np.abs(error).sum(axis=0)
            baseline_absolute += np.abs(baseline_error).sum(axis=0)
            pred_state = prediction[:, 22:] >= thresholds
            true_state = target[:, 22:] >= thresholds
            base_state = baseline[:, 22:] >= thresholds
            tp += np.sum(pred_state & true_state, axis=0)
            fp += np.sum(pred_state & ~true_state, axis=0)
            fn += np.sum(~pred_state & true_state, axis=0)
            baseline_tp += np.sum(base_state & true_state, axis=0)
            baseline_fp += np.sum(base_state & ~true_state, axis=0)
            baseline_fn += np.sum(~base_state & true_state, axis=0)
            positives += np.sum(true_state, axis=0)
            point_count += len(target)
    mse = squared / point_count
    baseline_mse = baseline_squared / point_count
    mae = absolute / point_count
    baseline_mae = baseline_absolute / point_count
    relative_mse = float(np.mean(mse[:22]) / np.mean(baseline_mse[:22]))
    denom = 2 * tp + fp + fn
    base_denom = 2 * baseline_tp + baseline_fp + baseline_fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(5), where=denom > 0)
    base_f1 = np.divide(
        2 * baseline_tp, base_denom, out=np.zeros(5), where=base_denom > 0
    )
    active = (positives > 0) & (positives < point_count)
    active_f1 = float(np.mean(f1[active])) if np.any(active) else 0.0
    active_base_f1 = float(np.mean(base_f1[active])) if np.any(active) else 0.0
    objective = 0.7 * relative_mse + 0.3 * (1 - active_f1)
    by_feature = {
        name: {
            "standardized_mae": float(mae[i]),
            "persistence_standardized_mae": float(baseline_mae[i]),
            "standardized_mse": float(mse[i]),
            "persistence_standardized_mse": float(baseline_mse[i]),
            **(
                {
                    "f1": float(f1[i - 22]),
                    "persistence_f1": float(base_f1[i - 22]),
                }
                if i >= 22 else {}
            ),
        }
        for i, name in enumerate(contract.names)
    }
    return {
        "objective_selected_27": objective,
        "forecast_points": int(point_count),
        "continuous_relative_mse_vs_persistence": relative_mse,
        "continuous_skill_vs_persistence": 1 - relative_mse,
        "continuous_standardized_mae": float(np.mean(mae[:22])),
        "boolean_active_macro_f1": active_f1,
        "persistence_boolean_active_macro_f1": active_base_f1,
        "active_boolean_feature_count": int(np.sum(active)),
        "metrics_by_feature": by_feature,
    }


def train_fold_27(
    repository: ExperimentRepository,
    contract: TargetContract,
    fold_data: FoldData27,
    library_root: Path,
    params: dict,
    output_dir: Path,
    epochs: int,
    patience: int,
    num_workers: int,
    seed: int,
    use_amp: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("133-to-27 training requires CUDA")
    model = SelectedOutputCrossformer(library_root, params, contract.input_indices).to(device)
    criterion = GroupedTargetHuberLoss(
        float(params["huber_delta"]), float(params["continuous_loss_weight"])
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loader = make_loader(
        fold_data.train_dataset, int(params["batch_size"]), True, num_workers, seed
    )
    validation_loader = make_loader(
        fold_data.validation_dataset, int(params["batch_size"]), False, num_workers, seed
    )
    history, best_metrics = [], None
    best_objective, best_epoch, stale = math.inf, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        sums = np.zeros(3, dtype=np.float64)
        batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(batch_x, None, None, None)
                components = criterion.forward_with_components(output, batch_y)
                loss = components[0]
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            sums += [float(value.detach().item()) for value in components]
            batches += 1
        scheduler.step()
        metrics = evaluate_model_27(
            model, validation_loader, contract, fold_data.mean, fold_data.scale,
            device, use_amp
        )
        objective = float(metrics["objective_selected_27"])
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(sums[0] / batches),
            "train_continuous_loss": float(sums[1] / batches),
            "train_boolean_loss": float(sums[2] / batches),
            **metrics,
        }
        history.append(row)
        print(
            f"{fold_data.name} epoch={epoch} objective={objective:.6f} "
            f"rel_mse={metrics['continuous_relative_mse_vs_persistence']:.6f} "
            f"bool_f1={metrics['boolean_active_macro_f1']:.6f}", flush=True
        )
        if objective < best_objective:
            best_objective, best_epoch, best_metrics, stale = objective, epoch, metrics, 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            stale += 1
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if stale >= patience:
            break
    result = {
        "fold": fold_data.name,
        "train_devices": list(fold_data.train_devices),
        "validation_devices": list(fold_data.validation_devices),
        "train_windows": len(fold_data.train_dataset),
        "validation_windows": len(fold_data.validation_dataset),
        "train_windows_by_device": fold_data.train_dataset.selected_counts,
        "validation_windows_by_device": fold_data.validation_dataset.selected_counts,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model, optimizer, amp_scaler, train_loader, validation_loader
    torch.cuda.empty_cache()
    return result

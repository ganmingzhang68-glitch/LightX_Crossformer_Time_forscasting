from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evenly_spaced_indices(count: int, limit: int | None) -> np.ndarray:
    if limit is None or limit <= 0 or count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


class ExperimentRepository:
    def __init__(self, data_root: Path, include_external: bool = False):
        self.data_root = data_root.resolve()
        manifest_path = self.data_root / "feature_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.feature_names = list(self.manifest["feature_order"])
        self.old_features = list(self.manifest["old_features"])
        self.added_features = list(self.manifest["added_features"])
        self.devices = list(self.manifest["training_devices"])
        if include_external:
            self.devices.append(self.manifest["external_test_device"])
        if len(self.feature_names) != 133 or len(self.old_features) != 128:
            raise ValueError("Expected a 133-channel manifest with 128 primary channels")
        self.boolean_indices = np.asarray(
            [
                i
                for i, feature in enumerate(self.feature_names)
                if feature in set(self.manifest["boolean_features"])
            ],
            dtype=np.int64,
        )
        self.continuous_indices = np.asarray(
            [i for i in range(len(self.feature_names)) if i not in set(self.boolean_indices)],
            dtype=np.int64,
        )
        self.core_boolean_indices = self.boolean_indices[self.boolean_indices < 128]
        self.core_continuous_indices = self.continuous_indices[self.continuous_indices < 128]
        self.auxiliary_indices = np.arange(128, 133, dtype=np.int64)
        if not len(self.core_boolean_indices) or not len(self.core_continuous_indices):
            raise ValueError("Primary 128 channels must include continuous and boolean fields")
        self.raw_values: dict[str, np.ndarray] = {}
        self.windows: dict[str, pd.DataFrame] = {}
        for device in self.devices:
            device_dir = self.data_root / "devices" / device
            frame = pd.read_parquet(device_dir / "rows.parquet", columns=self.feature_names)
            values = frame.to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite feature values for {device}")
            windows = pd.read_parquet(
                device_dir / "windows.parquet",
                columns=[
                    "past_start_row",
                    "past_end_row_exclusive",
                    "future_start_row",
                    "future_end_row_exclusive",
                ],
            )
            self.raw_values[device] = values
            self.windows[device] = windows

    def load_scaler(self, fold_name: str) -> tuple[np.ndarray, np.ndarray]:
        scaler_path = self.data_root / "folds" / fold_name / "scaler.npz"
        with np.load(scaler_path) as scaler:
            mean = scaler["mean"].astype(np.float32)
            scale = scaler["scale"].astype(np.float32)
        if mean.shape != (133,) or scale.shape != (133,) or np.any(scale <= 0):
            raise ValueError(f"Invalid scaler for {fold_name}")
        return mean, scale


class WindowDataset(Dataset):
    def __init__(
        self,
        scaled_values: dict[str, np.ndarray],
        windows: dict[str, pd.DataFrame],
        devices: tuple[str, ...],
        max_windows_per_device: int | None,
        balance_devices: bool,
    ):
        if not devices:
            raise ValueError("WindowDataset requires at least one device")
        counts = {device: len(windows[device]) for device in devices}
        if min(counts.values()) <= 0:
            raise ValueError(f"A selected device has no windows: {counts}")
        target = min(counts.values()) if balance_devices and len(devices) > 1 else None
        if max_windows_per_device and max_windows_per_device > 0:
            target = min(target, max_windows_per_device) if target else max_windows_per_device

        data_arrays: list[np.ndarray] = []
        device_codes: list[np.ndarray] = []
        past_starts: list[np.ndarray] = []
        future_starts: list[np.ndarray] = []
        future_ends: list[np.ndarray] = []
        selected_counts: dict[str, int] = {}
        for code, device in enumerate(devices):
            frame = windows[device]
            positions = evenly_spaced_indices(len(frame), target)
            selected = frame.iloc[positions]
            data_arrays.append(scaled_values[device])
            device_codes.append(np.full(len(selected), code, dtype=np.int16))
            past_starts.append(selected.past_start_row.to_numpy(dtype=np.int64))
            future_starts.append(selected.future_start_row.to_numpy(dtype=np.int64))
            future_ends.append(selected.future_end_row_exclusive.to_numpy(dtype=np.int64))
            selected_counts[device] = len(selected)
        self.data_arrays = data_arrays
        self.devices = devices
        self.device_codes = np.concatenate(device_codes)
        self.past_starts = np.concatenate(past_starts)
        self.future_starts = np.concatenate(future_starts)
        self.future_ends = np.concatenate(future_ends)
        self.selected_counts = selected_counts
        self.context_points = 150
        self.prediction_points = 60
        if not np.all(self.future_ends - self.future_starts == self.prediction_points):
            raise ValueError("Prepared windows do not have 60 forecast points")

    def __len__(self) -> int:
        return len(self.device_codes)

    def __getitem__(self, index: int):
        code = int(self.device_codes[index])
        past_start = int(self.past_starts[index])
        future_start = int(self.future_starts[index])
        future_end = int(self.future_ends[index])
        values = self.data_arrays[code]
        x = values[past_start : past_start + self.context_points]
        y = values[future_start:future_end]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class FoldData:
    name: str
    train_devices: tuple[str, ...]
    validation_devices: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    scaled_values: dict[str, np.ndarray]
    train_dataset: WindowDataset
    validation_dataset: WindowDataset


def prepare_fold_data(
    repository: ExperimentRepository,
    fold_name: str,
    max_train_windows_per_device: int | None,
    max_validation_windows_per_device: int | None,
) -> FoldData:
    fold = repository.manifest["folds"][fold_name]
    train_devices = tuple(fold["train_devices"])
    validation_devices = tuple(fold["validation_devices"])
    if not validation_devices:
        raise ValueError(f"{fold_name} has no validation device")
    mean, scale = repository.load_scaler(fold_name)
    required_devices = (*train_devices, *validation_devices)
    scaled = {
        device: ((repository.raw_values[device] - mean) / scale).astype(np.float32)
        for device in required_devices
    }
    train_dataset = WindowDataset(
        scaled,
        repository.windows,
        train_devices,
        max_train_windows_per_device,
        balance_devices=True,
    )
    validation_dataset = WindowDataset(
        scaled,
        repository.windows,
        validation_devices,
        max_validation_windows_per_device,
        balance_devices=False,
    )
    return FoldData(
        name=fold_name,
        train_devices=train_devices,
        validation_devices=validation_devices,
        mean=mean,
        scale=scale,
        scaled_values=scaled,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )


class CoreAuxiliaryHuberLoss(nn.Module):
    def __init__(
        self,
        core_continuous_indices: np.ndarray,
        core_boolean_indices: np.ndarray,
        auxiliary_indices: np.ndarray,
        huber_delta: float,
        continuous_weight: float,
        auxiliary_weight: float,
    ):
        super().__init__()
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0 <= auxiliary_weight < 1:
            raise ValueError("auxiliary_weight must be in [0, 1)")
        if not 0 < continuous_weight < 1:
            raise ValueError("continuous_weight must be in (0, 1)")
        self.core_continuous_indices = tuple(int(i) for i in core_continuous_indices)
        self.core_boolean_indices = tuple(int(i) for i in core_boolean_indices)
        self.auxiliary_indices = tuple(int(i) for i in auxiliary_indices)
        self.huber_delta = float(huber_delta)
        self.continuous_weight = float(continuous_weight)
        self.boolean_weight = 1.0 - self.continuous_weight
        self.auxiliary_weight = float(auxiliary_weight)

    def forward_with_components(self, prediction: torch.Tensor, target: torch.Tensor):
        elementwise = F.huber_loss(
            prediction, target, reduction="none", delta=self.huber_delta
        )
        continuous = elementwise[..., list(self.core_continuous_indices)].mean()
        boolean = elementwise[..., list(self.core_boolean_indices)].mean()
        auxiliary = elementwise[..., list(self.auxiliary_indices)].mean()
        core = self.continuous_weight * continuous + self.boolean_weight * boolean
        total = (1.0 - self.auxiliary_weight) * core + self.auxiliary_weight * auxiliary
        return total, core, continuous, boolean, auxiliary

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return self.forward_with_components(prediction, target)[0]


def build_model(library_root: Path, params: dict):
    library_root = library_root.resolve()
    if str(library_root) not in sys.path:
        sys.path.insert(0, str(library_root))
    from models.Crossformer import Model

    d_model = int(params["d_model"])
    n_heads = int(params["n_heads"])
    if d_model % n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    args = SimpleNamespace(
        enc_in=133,
        seq_len=150,
        pred_len=60,
        task_name="long_term_forecast",
        d_model=d_model,
        n_heads=n_heads,
        e_layers=int(params["e_layers"]),
        d_ff=int(params["d_ff"]),
        factor=int(params["factor"]),
        dropout=float(params["dropout"]),
        crossformer_seg_len=int(params["crossformer_seg_len"]),
        crossformer_win_size=2,
        use_revin=True,
        revin_affine=bool(params["revin_affine"]),
        residual_forecast=True,
    )
    return Model(args).float()


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    repository: ExperimentRepository,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> dict:
    feature_count = len(repository.feature_names)
    squared_sum = np.zeros(feature_count, dtype=np.float64)
    baseline_squared_sum = np.zeros(feature_count, dtype=np.float64)
    absolute_sum = np.zeros(feature_count, dtype=np.float64)
    baseline_absolute_sum = np.zeros(feature_count, dtype=np.float64)
    boolean_tp = np.zeros(feature_count, dtype=np.int64)
    boolean_fp = np.zeros(feature_count, dtype=np.int64)
    boolean_fn = np.zeros(feature_count, dtype=np.int64)
    baseline_tp = np.zeros(feature_count, dtype=np.int64)
    baseline_fp = np.zeros(feature_count, dtype=np.int64)
    baseline_fn = np.zeros(feature_count, dtype=np.int64)
    boolean_positive = np.zeros(feature_count, dtype=np.int64)
    point_count = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(batch_x, None, None, None)
            prediction = output.float().cpu().numpy().reshape(-1, feature_count)
            target = batch_y.float().cpu().numpy().reshape(-1, feature_count)
            baseline = (
                batch_x[:, -1:, :]
                .expand(-1, batch_y.shape[1], -1)
                .float()
                .cpu()
                .numpy()
                .reshape(-1, feature_count)
            )
            error = prediction.astype(np.float64) - target.astype(np.float64)
            baseline_error = baseline.astype(np.float64) - target.astype(np.float64)
            squared_sum += np.square(error).sum(axis=0)
            baseline_squared_sum += np.square(baseline_error).sum(axis=0)
            absolute_sum += np.abs(error).sum(axis=0)
            baseline_absolute_sum += np.abs(baseline_error).sum(axis=0)

            idx = repository.core_boolean_indices
            thresholds = (0.5 - scaler_mean[idx]) / scaler_scale[idx]
            predicted_state = prediction[:, idx] >= thresholds
            true_state = target[:, idx] >= thresholds
            baseline_state = baseline[:, idx] >= thresholds
            boolean_tp[idx] += np.sum(predicted_state & true_state, axis=0)
            boolean_fp[idx] += np.sum(predicted_state & ~true_state, axis=0)
            boolean_fn[idx] += np.sum(~predicted_state & true_state, axis=0)
            baseline_tp[idx] += np.sum(baseline_state & true_state, axis=0)
            baseline_fp[idx] += np.sum(baseline_state & ~true_state, axis=0)
            baseline_fn[idx] += np.sum(~baseline_state & true_state, axis=0)
            boolean_positive[idx] += np.sum(true_state, axis=0)
            point_count += len(target)
    if point_count == 0:
        raise RuntimeError("No validation points were evaluated")

    mse = squared_sum / point_count
    baseline_mse = baseline_squared_sum / point_count
    mae = absolute_sum / point_count
    baseline_mae = baseline_absolute_sum / point_count
    continuous = repository.core_continuous_indices
    model_continuous_mse = float(np.mean(mse[continuous]))
    baseline_continuous_mse = float(np.mean(baseline_mse[continuous]))
    relative_mse = (
        model_continuous_mse / baseline_continuous_mse
        if baseline_continuous_mse > np.finfo(np.float64).eps
        else float("inf")
    )
    denominator = 2 * boolean_tp + boolean_fp + boolean_fn
    baseline_denominator = 2 * baseline_tp + baseline_fp + baseline_fn
    f1 = np.zeros(feature_count, dtype=np.float64)
    baseline_f1 = np.zeros(feature_count, dtype=np.float64)
    valid = denominator > 0
    valid_baseline = baseline_denominator > 0
    f1[valid] = 2 * boolean_tp[valid] / denominator[valid]
    baseline_f1[valid_baseline] = 2 * baseline_tp[valid_baseline] / baseline_denominator[valid_baseline]
    active = (
        (boolean_positive[repository.core_boolean_indices] > 0)
        & (boolean_positive[repository.core_boolean_indices] < point_count)
    )
    active_f1 = (
        float(np.mean(f1[repository.core_boolean_indices][active]))
        if np.any(active)
        else 0.0
    )
    active_baseline_f1 = (
        float(np.mean(baseline_f1[repository.core_boolean_indices][active]))
        if np.any(active)
        else 0.0
    )
    objective = 0.7 * relative_mse + 0.3 * (1.0 - active_f1)
    auxiliary = repository.auxiliary_indices
    per_auxiliary = {
        repository.feature_names[i]: {
            "standardized_mae": float(mae[i]),
            "persistence_standardized_mae": float(baseline_mae[i]),
            "standardized_mse": float(mse[i]),
            "persistence_standardized_mse": float(baseline_mse[i]),
        }
        for i in auxiliary
    }
    return {
        "objective_primary_128": float(objective),
        "forecast_points": int(point_count),
        "core_continuous_relative_mse_vs_persistence": float(relative_mse),
        "core_continuous_skill_vs_persistence": float(1.0 - relative_mse),
        "core_continuous_standardized_mae": float(np.mean(mae[continuous])),
        "core_boolean_active_macro_f1": active_f1,
        "persistence_core_boolean_active_macro_f1": active_baseline_f1,
        "active_core_boolean_feature_count": int(np.sum(active)),
        "auxiliary_standardized_mae": float(np.mean(mae[auxiliary])),
        "persistence_auxiliary_standardized_mae": float(np.mean(baseline_mae[auxiliary])),
        "auxiliary_by_feature": per_auxiliary,
    }


def train_fold(
    repository: ExperimentRepository,
    fold_data: FoldData,
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
    model = build_model(library_root, params).to(device)
    criterion = CoreAuxiliaryHuberLoss(
        repository.core_continuous_indices,
        repository.core_boolean_indices,
        repository.auxiliary_indices,
        huber_delta=float(params["huber_delta"]),
        continuous_weight=float(params["continuous_loss_weight"]),
        auxiliary_weight=float(params["auxiliary_loss_weight"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loader = make_loader(
        fold_data.train_dataset,
        int(params["batch_size"]),
        True,
        num_workers,
        seed,
    )
    validation_loader = make_loader(
        fold_data.validation_dataset,
        int(params["batch_size"]),
        False,
        num_workers,
        seed,
    )
    history = []
    best_objective = math.inf
    best_metrics: dict | None = None
    best_epoch = 0
    stale_epochs = 0
    checkpoint = output_dir / "best_model.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        sums = np.zeros(5, dtype=np.float64)
        batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(batch_x, None, None, None)
                components = criterion.forward_with_components(output, batch_y)
                loss = components[0]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            sums += np.asarray([float(value.detach().item()) for value in components])
            batches += 1
        scheduler.step()
        metrics = evaluate_model(
            model,
            validation_loader,
            repository,
            fold_data.mean,
            fold_data.scale,
            device,
            use_amp,
        )
        objective = float(metrics["objective_primary_128"])
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_total_loss": float(sums[0] / batches),
            "train_core_loss": float(sums[1] / batches),
            "train_core_continuous_loss": float(sums[2] / batches),
            "train_core_boolean_loss": float(sums[3] / batches),
            "train_auxiliary_loss": float(sums[4] / batches),
            **metrics,
        }
        history.append(row)
        print(
            f"{fold_data.name} epoch={epoch} objective={objective:.6f} "
            f"rel_mse={metrics['core_continuous_relative_mse_vs_persistence']:.6f} "
            f"bool_f1={metrics['core_boolean_active_macro_f1']:.6f}",
            flush=True,
        )
        if objective < best_objective:
            best_objective = objective
            best_metrics = metrics
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if stale_epochs >= patience:
            break
    if best_metrics is None:
        raise RuntimeError("Fold training produced no validation metrics")
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
    del model, optimizer, scaler, train_loader, validation_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

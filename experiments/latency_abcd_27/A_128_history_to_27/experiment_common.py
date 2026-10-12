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


FOLDS = ("fold_a", "fold_b", "fold_c")
CONTEXT_POINTS = 150
PREDICTION_POINTS = 60
OUTPUT_CHANNELS = 27


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


class ExperimentContract:
    def __init__(self, spec_path: Path, data_root: Path):
        self.spec_path = spec_path.resolve()
        self.spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        self.data_root = data_root.resolve()
        self.manifest = json.loads(
            (self.data_root / "feature_manifest.json").read_text(encoding="utf-8")
        )
        self.source_features = list(self.manifest["feature_order"])
        self.old_features = list(self.manifest["old_features"])
        expected = int(self.spec["expected_source_feature_count"])
        if len(self.source_features) != expected:
            raise ValueError(
                f"{self.spec['experiment_id']} expects {expected} source fields, "
                f"got {len(self.source_features)}"
            )
        policy = self.spec["dynamic_feature_policy"]
        if policy == "old_features":
            dynamic = list(self.old_features)
        elif policy == "old_features_plus_named":
            dynamic = [*self.old_features, *self.spec["dynamic_extra_features"]]
        else:
            raise ValueError(f"Unsupported dynamic_feature_policy: {policy}")
        conditions = list(self.spec.get("condition_features", []))
        if len(set(dynamic + conditions)) != len(dynamic) + len(conditions):
            raise ValueError("Dynamic and condition fields must be disjoint and unique")
        missing = [name for name in dynamic + conditions if name not in self.source_features]
        if missing:
            raise ValueError(f"Experiment fields missing from source manifest: {missing}")
        target_path = self.spec_path.parent / self.spec["target_contract"]
        target_doc = json.loads(target_path.read_text(encoding="utf-8"))
        self.continuous_targets = list(target_doc["continuous_features"])
        self.boolean_targets = list(target_doc["boolean_features"])
        self.target_features = self.continuous_targets + self.boolean_targets
        if len(self.continuous_targets) != 22 or len(self.boolean_targets) != 5:
            raise ValueError("Target contract must contain 22 continuous and 5 boolean fields")
        if len(set(self.target_features)) != OUTPUT_CHANNELS:
            raise ValueError("Target contract must contain 27 unique fields")
        if any(name not in dynamic for name in self.target_features):
            raise ValueError("Every target must be present in the dynamic historical branch")
        self.dynamic_features = dynamic
        self.condition_features = conditions
        self.categorical_condition_features = list(
            self.spec.get("categorical_condition_features", [])
        )
        if any(name not in conditions for name in self.categorical_condition_features):
            raise ValueError("Categorical conditions must be condition fields")
        self.dynamic_source_indices = np.asarray(
            [self.source_features.index(name) for name in dynamic], dtype=np.int64
        )
        self.condition_source_indices = np.asarray(
            [self.source_features.index(name) for name in conditions], dtype=np.int64
        )
        self.target_source_indices = np.asarray(
            [self.source_features.index(name) for name in self.target_features], dtype=np.int64
        )
        self.target_dynamic_indices = np.asarray(
            [dynamic.index(name) for name in self.target_features], dtype=np.int64
        )
        self.condition_categorical_positions = {
            conditions.index(name): tuple(self.spec["categorical_values"][name])
            for name in self.categorical_condition_features
        }
        self.condition_continuous_positions = tuple(
            i for i in range(len(conditions))
            if i not in self.condition_categorical_positions
        )
        self.input_field_count = len(dynamic) + len(conditions)

    def describe(self) -> dict:
        return {
            "experiment_id": self.spec["experiment_id"],
            "source_feature_count": len(self.source_features),
            "total_input_fields": self.input_field_count,
            "dynamic_history_fields": len(self.dynamic_features),
            "condition_fields": len(self.condition_features),
            "output_fields": len(self.target_features),
            "dynamic_feature_order": self.dynamic_features,
            "condition_feature_order": self.condition_features,
            "target_feature_order": self.target_features,
            "restricted_conditions": self.spec.get("restricted_conditions", {}),
        }


class ExperimentRepository:
    def __init__(self, contract: ExperimentContract, include_external: bool = False):
        self.contract = contract
        self.manifest = contract.manifest
        devices = list(self.manifest["training_devices"])
        if include_external:
            devices.append(self.manifest["external_test_device"])
        self.devices = devices
        self.raw_values: dict[str, np.ndarray] = {}
        self.windows: dict[str, pd.DataFrame] = {}
        for device in devices:
            root = contract.data_root / "devices" / device
            frame = pd.read_parquet(root / "rows.parquet", columns=contract.source_features)
            values = frame.to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite source values for {device}")
            windows = pd.read_parquet(
                root / "windows.parquet",
                columns=["past_start_row", "past_end_row_exclusive",
                         "future_start_row", "future_end_row_exclusive"],
            )
            if not np.all(
                windows.future_end_row_exclusive.to_numpy()
                - windows.future_start_row.to_numpy() == PREDICTION_POINTS
            ):
                raise ValueError(f"Unexpected forecast length for {device}")
            self.raw_values[device] = values
            self.windows[device] = windows

    def load_scaler(self, fold_name: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.contract.data_root / "folds" / fold_name / "scaler.npz") as data:
            mean = data["mean"].astype(np.float32)
            scale = data["scale"].astype(np.float32)
        expected = (len(self.contract.source_features),)
        if mean.shape != expected or scale.shape != expected or np.any(scale <= 0):
            raise ValueError(f"Invalid source scaler for {fold_name}")
        return mean, scale


class ExperimentWindowDataset(Dataset):
    def __init__(self, scaled_values: dict[str, np.ndarray], windows: dict[str, pd.DataFrame],
                 devices: tuple[str, ...], contract: ExperimentContract,
                 max_windows_per_device: int | None, balance_devices: bool):
        if not devices:
            raise ValueError("At least one device is required")
        counts = {name: len(windows[name]) for name in devices}
        if min(counts.values()) <= 0:
            raise ValueError(f"Selected device has no windows: {counts}")
        limit = min(counts.values()) if balance_devices and len(devices) > 1 else None
        if max_windows_per_device and max_windows_per_device > 0:
            limit = min(limit, max_windows_per_device) if limit else max_windows_per_device
        arrays, codes, past, start, end = [], [], [], [], []
        self.selected_counts = {}
        for code, name in enumerate(devices):
            chosen = windows[name].iloc[evenly_spaced_indices(counts[name], limit)]
            arrays.append(scaled_values[name])
            codes.append(np.full(len(chosen), code, dtype=np.int16))
            past.append(chosen.past_start_row.to_numpy(dtype=np.int64))
            start.append(chosen.future_start_row.to_numpy(dtype=np.int64))
            end.append(chosen.future_end_row_exclusive.to_numpy(dtype=np.int64))
            self.selected_counts[name] = len(chosen)
        self.arrays = arrays
        self.codes = np.concatenate(codes)
        self.past = np.concatenate(past)
        self.start = np.concatenate(start)
        self.end = np.concatenate(end)
        self.dynamic_indices = contract.dynamic_source_indices
        self.condition_indices = contract.condition_source_indices
        self.target_indices = contract.target_source_indices

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, index: int):
        values = self.arrays[int(self.codes[index])]
        past = int(self.past[index])
        start, end = int(self.start[index]), int(self.end[index])
        x_dynamic = values[past:past + CONTEXT_POINTS, self.dynamic_indices]
        if len(self.condition_indices):
            x_condition = values[past:past + CONTEXT_POINTS, self.condition_indices]
        else:
            x_condition = np.empty((CONTEXT_POINTS, 0), dtype=np.float32)
        y = values[start:end, self.target_indices]
        expected = (CONTEXT_POINTS, len(self.dynamic_indices))
        if x_dynamic.shape != expected or y.shape != (PREDICTION_POINTS, OUTPUT_CHANNELS):
            raise RuntimeError(f"Unexpected sample shapes: {x_dynamic.shape}, {y.shape}")
        return torch.from_numpy(x_dynamic), torch.from_numpy(x_condition), torch.from_numpy(y)


@dataclass
class FoldData:
    name: str
    train_devices: tuple[str, ...]
    validation_devices: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    train_dataset: ExperimentWindowDataset
    validation_dataset: ExperimentWindowDataset


def prepare_fold_data(repository: ExperimentRepository, fold_name: str,
                      max_train: int | None, max_validation: int | None) -> FoldData:
    split = repository.manifest["folds"][fold_name]
    train = tuple(split["train_devices"])
    validation = tuple(split["validation_devices"])
    mean, scale = repository.load_scaler(fold_name)
    scaled = {
        name: ((repository.raw_values[name] - mean) / scale).astype(np.float32)
        for name in (*train, *validation)
    }
    return FoldData(
        fold_name, train, validation, mean, scale,
        ExperimentWindowDataset(scaled, repository.windows, train, repository.contract,
                                max_train, True),
        ExperimentWindowDataset(scaled, repository.windows, validation, repository.contract,
                                max_validation, False),
    )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, generator=generator)


def build_crossformer(library_root: Path, params: dict, enc_in: int) -> nn.Module:
    root = str(library_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.Crossformer import Model
    config = SimpleNamespace(
        enc_in=enc_in, seq_len=CONTEXT_POINTS, pred_len=PREDICTION_POINTS,
        task_name="long_term_forecast", d_model=int(params["d_model"]),
        n_heads=int(params["n_heads"]), e_layers=int(params["e_layers"]),
        d_ff=int(params["d_ff"]), factor=int(params["factor"]),
        dropout=float(params["dropout"]),
        crossformer_seg_len=int(params["crossformer_seg_len"]),
        crossformer_win_size=2, use_revin=True,
        revin_affine=bool(params["revin_affine"]), residual_forecast=True,
    )
    return Model(config).float()


class ConditionEncoder(nn.Module):
    def __init__(self, contract: ExperimentContract, params: dict,
                 source_mean: np.ndarray, source_scale: np.ndarray):
        super().__init__()
        hidden = int(params["condition_hidden"])
        self.continuous_positions = contract.condition_continuous_positions
        self.category_positions = tuple(contract.condition_categorical_positions)
        self.embeddings = nn.ModuleDict()
        self.vocabularies: dict[int, tuple[float, ...]] = {}
        embedding_width = int(params.get("condition_embedding_dim", 4))
        for position, values in contract.condition_categorical_positions.items():
            self.embeddings[str(position)] = nn.Embedding(len(values) + 1, embedding_width)
            self.vocabularies[position] = values
        condition_mean = source_mean[contract.condition_source_indices]
        condition_scale = source_scale[contract.condition_source_indices]
        self.register_buffer("condition_mean", torch.as_tensor(condition_mean))
        self.register_buffer("condition_scale", torch.as_tensor(condition_scale))
        width = len(self.continuous_positions) + embedding_width * len(self.category_positions)
        if width <= 0:
            raise ValueError("ConditionEncoder needs at least one condition field")
        layers = int(params.get("condition_layers", 1))
        dropout = float(params.get("condition_dropout", 0.0)) if layers > 1 else 0.0
        self.gru = nn.GRU(width, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(float(params.get("condition_dropout", 0.0))),
            nn.Linear(hidden, PREDICTION_POINTS * OUTPUT_CHANNELS),
        )
        self.gate = nn.Parameter(torch.tensor(float(params.get("condition_gate_init", 0.1))))

    def forward(self, scaled: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.continuous_positions:
            parts.append(scaled[..., list(self.continuous_positions)])
        raw = scaled * self.condition_scale + self.condition_mean
        for position in self.category_positions:
            values = self.vocabularies[position]
            indices = torch.full(raw.shape[:2], len(values), dtype=torch.long, device=raw.device)
            for index, value in enumerate(values):
                indices = torch.where(torch.isclose(raw[..., position],
                    torch.as_tensor(value, device=raw.device, dtype=raw.dtype), atol=1e-4),
                    index, indices)
            parts.append(self.embeddings[str(position)](indices))
        sequence = torch.cat(parts, dim=-1)
        _, hidden = self.gru(sequence)
        correction = self.head(hidden[-1]).reshape(-1, PREDICTION_POINTS, OUTPUT_CHANNELS)
        return torch.tanh(self.gate) * correction


class ExperimentModel(nn.Module):
    def __init__(self, contract: ExperimentContract, library_root: Path, params: dict,
                 source_mean: np.ndarray, source_scale: np.ndarray):
        super().__init__()
        self.backbone = build_crossformer(library_root, params, len(contract.dynamic_features))
        self.register_buffer("target_dynamic_indices",
            torch.as_tensor(contract.target_dynamic_indices, dtype=torch.long))
        self.condition_encoder = (
            ConditionEncoder(contract, params, source_mean, source_scale)
            if contract.condition_features else None
        )

    def forward(self, dynamic: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        full = self.backbone(dynamic, None, None, None)
        selected = torch.index_select(full, -1, self.target_dynamic_indices)
        if self.condition_encoder is not None:
            selected = selected + self.condition_encoder(condition)
        return selected


class GroupedHuberLoss(nn.Module):
    def __init__(self, delta: float, continuous_weight: float):
        super().__init__()
        self.delta = float(delta)
        self.continuous_weight = float(continuous_weight)

    def forward_with_components(self, prediction: torch.Tensor, target: torch.Tensor):
        values = F.huber_loss(prediction, target, reduction="none", delta=self.delta)
        continuous = values[..., :22].mean()
        boolean = values[..., 22:].mean()
        total = self.continuous_weight * continuous + (1 - self.continuous_weight) * boolean
        return total, continuous, boolean

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return self.forward_with_components(prediction, target)[0]


def evaluate(model: nn.Module, loader: DataLoader, contract: ExperimentContract,
             mean: np.ndarray, scale: np.ndarray, device: torch.device, use_amp: bool) -> dict:
    count = OUTPUT_CHANNELS
    squared = np.zeros(count); base_squared = np.zeros(count)
    absolute = np.zeros(count); base_absolute = np.zeros(count)
    tp = np.zeros(5, dtype=np.int64); fp = tp.copy(); fn = tp.copy()
    btp = tp.copy(); bfp = tp.copy(); bfn = tp.copy(); positives = tp.copy()
    points = 0
    target_mean = mean[contract.target_source_indices]
    target_scale = scale[contract.target_source_indices]
    thresholds = (0.5 - target_mean[22:]) / target_scale[22:]
    model.eval()
    with torch.no_grad():
        for dynamic, condition, target_tensor in loader:
            dynamic = dynamic.float().to(device, non_blocking=True)
            condition = condition.float().to(device, non_blocking=True)
            target_tensor = target_tensor.float().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(dynamic, condition)
            pred = output.float().cpu().numpy().reshape(-1, count)
            truth = target_tensor.cpu().numpy().reshape(-1, count)
            baseline_indices = torch.as_tensor(
                contract.target_dynamic_indices, dtype=torch.long, device=dynamic.device
            )
            baseline = torch.index_select(dynamic[:, -1:, :], -1, baseline_indices).expand(
                -1, target_tensor.shape[1], -1).cpu().numpy().reshape(-1, count)
            error = pred.astype(np.float64) - truth.astype(np.float64)
            base_error = baseline.astype(np.float64) - truth.astype(np.float64)
            squared += np.square(error).sum(0); base_squared += np.square(base_error).sum(0)
            absolute += np.abs(error).sum(0); base_absolute += np.abs(base_error).sum(0)
            ps, ts, bs = pred[:, 22:] >= thresholds, truth[:, 22:] >= thresholds, baseline[:, 22:] >= thresholds
            tp += np.sum(ps & ts, 0); fp += np.sum(ps & ~ts, 0); fn += np.sum(~ps & ts, 0)
            btp += np.sum(bs & ts, 0); bfp += np.sum(bs & ~ts, 0); bfn += np.sum(~bs & ts, 0)
            positives += np.sum(ts, 0); points += len(truth)
    mse, base_mse = squared / points, base_squared / points
    mae, base_mae = absolute / points, base_absolute / points
    relative = float(np.mean(mse[:22]) / np.mean(base_mse[:22]))
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(5), where=(2*tp+fp+fn)>0)
    bf1 = np.divide(2*btp, 2*btp+bfp+bfn, out=np.zeros(5), where=(2*btp+bfp+bfn)>0)
    active = (positives > 0) & (positives < points)
    macro = float(np.mean(f1[active])) if np.any(active) else 0.0
    base_macro = float(np.mean(bf1[active])) if np.any(active) else 0.0
    by_feature = {}
    for i, name in enumerate(contract.target_features):
        row = {"standardized_mae": float(mae[i]),
               "persistence_standardized_mae": float(base_mae[i]),
               "standardized_mse": float(mse[i]),
               "persistence_standardized_mse": float(base_mse[i])}
        if i >= 22:
            row.update(f1=float(f1[i-22]), persistence_f1=float(bf1[i-22]))
        by_feature[name] = row
    return {
        "objective_selected_27": 0.7 * relative + 0.3 * (1 - macro),
        "forecast_points": int(points),
        "continuous_relative_mse_vs_persistence": relative,
        "continuous_skill_vs_persistence": 1 - relative,
        "continuous_standardized_mae": float(np.mean(mae[:22])),
        "boolean_active_macro_f1": macro,
        "persistence_boolean_active_macro_f1": base_macro,
        "active_boolean_feature_count": int(np.sum(active)),
        "metrics_by_feature": by_feature,
    }


def train_fold(repository: ExperimentRepository, fold: FoldData, library_root: Path,
               params: dict, output: Path, epochs: int, patience: int,
               workers: int, seed: int, use_amp: bool) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training requires CUDA")
    model = ExperimentModel(repository.contract, library_root, params, fold.mean, fold.scale).to(device)
    loss_fn = GroupedHuberLoss(params["huber_delta"], params["continuous_loss_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loader = make_loader(fold.train_dataset, params["batch_size"], True, workers, seed)
    val_loader = make_loader(fold.validation_dataset, params["batch_size"], False, workers, seed)
    history=[]; best=math.inf; best_epoch=0; best_metrics=None; stale=0
    for epoch in range(1, epochs+1):
        model.train(); sums=np.zeros(3); batches=0
        for dynamic, condition, target in train_loader:
            dynamic=dynamic.float().to(device,non_blocking=True)
            condition=condition.float().to(device,non_blocking=True)
            target=target.float().to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=use_amp):
                components=loss_fn.forward_with_components(model(dynamic,condition),target)
            scaler.scale(components[0]).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            sums += [float(x.detach()) for x in components]; batches += 1
        scheduler.step()
        metrics=evaluate(model,val_loader,repository.contract,fold.mean,fold.scale,device,use_amp)
        objective=float(metrics["objective_selected_27"])
        row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],
             "train_loss":float(sums[0]/batches),"train_continuous_loss":float(sums[1]/batches),
             "train_boolean_loss":float(sums[2]/batches),**metrics}
        history.append(row)
        print(f"{fold.name} epoch={epoch} objective={objective:.6f} rel_mse={metrics['continuous_relative_mse_vs_persistence']:.6f} bool_f1={metrics['boolean_active_macro_f1']:.6f}",flush=True)
        if objective < best:
            best,best_epoch,best_metrics,stale=objective,epoch,metrics,0
            torch.save(model.state_dict(),output/"best_model.pth")
        else: stale += 1
        (output/"history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
        if stale >= patience: break
    result={"fold":fold.name,"train_devices":list(fold.train_devices),
            "validation_devices":list(fold.validation_devices),"train_windows":len(fold.train_dataset),
            "validation_windows":len(fold.validation_dataset),
            "train_windows_by_device":fold.train_dataset.selected_counts,
            "validation_windows_by_device":fold.validation_dataset.selected_counts,
            "best_epoch":best_epoch,"best_metrics":best_metrics}
    (output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    del model,optimizer,scaler,train_loader,val_loader
    torch.cuda.empty_cache()
    return result

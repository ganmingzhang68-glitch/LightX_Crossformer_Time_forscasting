from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crossformer import Crossformer, CrossformerConfig


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def choose_time_column(frame: pd.DataFrame) -> str:
    for column in ("time_beijing", "time_utc", "time"):
        if column in frame.columns:
            return column
    raise ValueError("Input must contain one timestamp column: time_beijing, time_utc, or time")


def contiguous_segments(frame: pd.DataFrame, time_column: str):
    timestamps = pd.to_datetime(frame[time_column], utc=True)
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    duplicate = timestamps.duplicated(keep="last")
    frame = frame.loc[~duplicate].reset_index(drop=True)
    timestamps = timestamps.loc[~duplicate].reset_index(drop=True)
    labels = timestamps.diff().dt.total_seconds().ne(2).cumsum()
    for _, indices in frame.groupby(labels, sort=True).groups.items():
        yield frame.loc[indices].reset_index(drop=True), timestamps.loc[indices].reset_index(drop=True)


def make_windows(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    features: list[str],
    sequence_length: int,
    prediction_length: int,
    stride: int,
    mode: str,
):
    values = frame[features].to_numpy(dtype=np.float32, copy=True)
    if len(values) < sequence_length:
        return []
    if mode == "latest":
        starts = [len(values) - sequence_length]
    else:
        final_start = len(values) - sequence_length - prediction_length
        starts = range(0, final_start + 1, stride) if final_start >= 0 else []
    windows = []
    for start in starts:
        end = start + sequence_length
        targets = None
        if end + prediction_length <= len(values):
            targets = values[end : end + prediction_length]
        windows.append((values[start:end], timestamps.iloc[end - 1], targets))
    return windows


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the trained LightX Crossformer on another device.")
    parser.add_argument("--input", type=Path, required=True, help="Prepared wide CSV or Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Prediction CSV or Parquet")
    parser.add_argument("--mode", choices=("latest", "rolling"), default="latest")
    parser.add_argument("--stride", type=int, default=5, help="Rolling window stride in 2-second points")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--include-target", action="store_true", help="Include actual values for rolling evaluation")
    parser.add_argument("--checkpoint", type=Path, default=root / "artifacts/checkpoint.pth")
    parser.add_argument("--manifest", type=Path, default=root / "artifacts/feature_manifest.json")
    parser.add_argument("--scaler", type=Path, default=root / "artifacts/scaler.json")
    parser.add_argument("--config", type=Path, default=root / "artifacts/model_config.json")
    args = parser.parse_args()
    if args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("stride and batch-size must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scaler = json.loads(args.scaler.read_text(encoding="utf-8"))
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    features = list(manifest["feature_order"])
    if features != list(scaler["feature_order"]):
        raise ValueError("Scaler and feature manifest orders differ")
    boolean_features = set(manifest["boolean_features"])
    boolean_indices = np.asarray(
        [index for index, feature in enumerate(features) if feature in boolean_features],
        dtype=np.int64,
    )
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    if len(mean) != len(features) or len(scale) != len(features) or np.any(scale <= 0):
        raise ValueError("Invalid scaler statistics")
    frame = load_frame(args.input)
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required features: {missing}")
    if not np.isfinite(frame[features].to_numpy(dtype=np.float32, copy=False)).all():
        raise ValueError("Input contains NaN or infinite feature values; run prepare_influx.py first")
    time_column = choose_time_column(frame)
    config = CrossformerConfig.from_dict(raw_config)
    if config.enc_in != len(features):
        raise ValueError("Model feature count differs from feature manifest")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = Crossformer(config)
    state = load_checkpoint(args.checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    all_windows = []
    for segment, timestamps in contiguous_segments(frame, time_column):
        all_windows.extend(make_windows(
            segment,
            timestamps,
            features,
            config.seq_len,
            config.pred_len,
            args.stride,
            args.mode,
        ))
    # Deployment mode emits one forecast only: the newest usable context in
    # the file, even when preprocessing produced several discontinuous parts.
    if args.mode == "latest" and all_windows:
        all_windows = all_windows[-1:]
    if not all_windows:
        raise RuntimeError(
            f"No usable context: each continuous segment needs at least {config.seq_len} rows"
        )
    output_records = []
    cadence = pd.Timedelta(seconds=int(raw_config["cadence_seconds"]))
    threshold = float(raw_config["boolean_threshold_physical"])
    for offset in range(0, len(all_windows), args.batch_size):
        batch = all_windows[offset : offset + args.batch_size]
        physical = np.stack([item[0] for item in batch])
        standardized = (physical - mean) / scale
        tensor = torch.from_numpy(standardized).to(device)
        with torch.inference_mode():
            prediction_standardized = model(tensor).float().cpu().numpy()
        prediction = prediction_standardized * scale + mean
        prediction[:, :, boolean_indices] = (
            prediction[:, :, boolean_indices] >= threshold
        ).astype(np.float32)
        for batch_index, (_, origin, targets) in enumerate(batch):
            for horizon in range(config.pred_len):
                record = {
                    "forecast_origin_time": origin,
                    "prediction_time": origin + cadence * (horizon + 1),
                    "horizon_step": horizon + 1,
                }
                record.update(zip(features, prediction[batch_index, horizon].tolist()))
                if args.include_target and targets is not None:
                    record.update({
                        f"actual__{feature}": float(targets[horizon, feature_index])
                        for feature_index, feature in enumerate(features)
                    })
                output_records.append(record)
    output = pd.DataFrame.from_records(output_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        output.to_csv(args.output, index=False)
    else:
        output.to_parquet(args.output, index=False, compression="zstd")
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(json.dumps({
        "output": str(args.output.resolve()),
        "mode": args.mode,
        "contexts": len(all_windows),
        "forecast_rows": len(output),
        "feature_count": len(features),
        "context_points": config.seq_len,
        "prediction_points": config.pred_len,
        "device": str(device),
        "checkpoint_sha256": checkpoint_sha256,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

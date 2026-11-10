from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEVICES = ("01-X450-1", "02-X450-2", "03-X450-3", "04-X450-4")
TRAIN_DEVICES = DEVICES[:3]
EXTERNAL_DEVICE = DEVICES[3]
ADDED_FEATURES = (
    "打印层数",
    "设松粉S",
    "设铺粉模式",
    "设前螺旋时长",
    "设后螺旋时长",
    "设风速",
)
EXCLUDED_DEVICE01_JOBS = {
    "01-X450-1_20260726T083710714006Z",
    "01-X450-1_20260727T115726163177Z",
    "01-X450-1_20260730T202943865451Z",
}
FOLDS = {
    "fold_a": {"train": ("01-X450-1", "02-X450-2"), "validation": ("03-X450-3",)},
    "fold_b": {"train": ("01-X450-1", "03-X450-3"), "validation": ("02-X450-2",)},
    "fold_c": {"train": ("02-X450-2", "03-X450-3"), "validation": ("01-X450-1",)},
    "final": {"train": TRAIN_DEVICES, "validation": ()},
}

CADENCE_SECONDS = 2
CONTEXT_POINTS = 150
GAP_POINTS = 15
PREDICTION_POINTS = 60
WINDOW_STRIDE_POINTS = 5
MAX_FILL_MISSING_POINTS = 1
MAX_CONTIGUOUS_DELTA_SECONDS = 4


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def load_old_rows(old_root: Path, old_features: list[str]) -> pd.DataFrame:
    required = [
        "time_utc",
        "time_beijing",
        "device_id",
        "segment_id",
        "observed_mask",
        *old_features,
    ]
    frames = []
    for split in ("train", "test", "validation"):
        path = old_root / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=required)
        frame["old_source_split"] = split
        frames.append(frame)
    old = pd.concat(frames, ignore_index=True)
    old = old.rename(
        columns={"segment_id": "old_segment_id", "observed_mask": "old_observed_mask"}
    )
    old["time_utc"] = pd.to_datetime(old["time_utc"], utc=True)
    duplicate = old.duplicated(["device_id", "time_utc"], keep=False)
    if duplicate.any():
        sample = old.loc[duplicate, ["device_id", "time_utc", "old_source_split"]].head()
        raise RuntimeError(f"Old processed splits overlap on device/time:\n{sample}")
    if old[old_features].isna().any().any():
        raise RuntimeError("Old processed rows contain null values in the 128 feature contract")
    return old.sort_values(["device_id", "time_utc"], kind="stable").reset_index(drop=True)


def regularize_new_device(path: Path, next_chunk_id: int) -> tuple[pd.DataFrame, list[dict], int]:
    required = [
        "time_utc",
        "device_id",
        "job_id",
        "source_file",
        "source_row",
        *ADDED_FEATURES,
    ]
    raw_device = pd.read_parquet(path, columns=required)
    records: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for job_id, raw in raw_device.groupby("job_id", sort=False):
        raw = raw.rename(columns={"time_utc": "new_source_time_utc"}).copy()
        raw["new_source_time_utc"] = pd.to_datetime(raw["new_source_time_utc"], utc=True)
        raw = raw.sort_values(["new_source_time_utc", "source_row"], kind="stable")
        source_duplicates = int(raw.duplicated("new_source_time_utc", keep="last").sum())
        raw = raw.drop_duplicates("new_source_time_utc", keep="last")
        raw["time_utc"] = raw["new_source_time_utc"].dt.ceil(f"{CADENCE_SECONDS}s")
        raw = raw.sort_values(["time_utc", "new_source_time_utc", "source_row"], kind="stable")
        grid_collisions = int(raw.duplicated("time_utc", keep="last").sum())
        raw = raw.drop_duplicates("time_utc", keep="last").copy()
        if raw.empty:
            continue
        if raw[list(ADDED_FEATURES)].isna().any().any():
            missing = raw[list(ADDED_FEATURES)].isna().sum()
            raise RuntimeError(
                f"New source job {job_id} has null selected fields: "
                f"{missing[missing > 0].to_dict()}"
            )
        deltas = raw["time_utc"].diff().dt.total_seconds()
        if deltas.dropna().le(0).any():
            raise RuntimeError(f"New source job {job_id} is not strictly increasing")
        chunk_labels = deltas.gt(MAX_CONTIGUOUS_DELTA_SECONDS).cumsum()
        for _, chunk in raw.groupby(chunk_labels, sort=True):
            chunk = chunk.set_index("time_utc").sort_index()
            grid = pd.date_range(
                chunk.index[0], chunk.index[-1], freq=f"{CADENCE_SECONDS}s", tz="UTC"
            )
            regular = chunk[
                ["new_source_time_utc", "source_row", *ADDED_FEATURES]
            ].reindex(grid)
            observed = regular[list(ADDED_FEATURES)].notna().all(axis=1)
            regular[list(ADDED_FEATURES)] = regular[list(ADDED_FEATURES)].ffill(
                limit=MAX_FILL_MISSING_POINTS
            )
            if regular[list(ADDED_FEATURES)].isna().any().any():
                raise RuntimeError(f"Unresolved new-source gap inside job {job_id}")
            frame = regular.reset_index(names="time_utc")
            frame.insert(1, "device_id", str(raw.device_id.iloc[0]))
            frame.insert(2, "job_id", str(job_id))
            frame.insert(3, "new_chunk_id", np.int32(next_chunk_id))
            frame.insert(4, "new_observed_mask", observed.to_numpy(dtype=bool))
            for feature in ADDED_FEATURES:
                frame[feature] = frame[feature].astype("float32")
            records.append(frame)
            summaries.append(
                {
                    "device_id": str(raw.device_id.iloc[0]),
                    "job_id": str(job_id),
                    "new_chunk_id": next_chunk_id,
                    "start_time_utc": frame.time_utc.iloc[0],
                    "end_time_utc": frame.time_utc.iloc[-1],
                    "regular_rows": int(len(frame)),
                    "inserted_rows": int((~observed).sum()),
                    "source_duplicate_rows": source_duplicates,
                    "grid_collision_rows": grid_collisions,
                }
            )
            next_chunk_id += 1
    if not records:
        raise RuntimeError(f"No new-source rows were prepared from {path}")
    return pd.concat(records, ignore_index=True), summaries, next_chunk_id


def merge_device(
    old_device: pd.DataFrame,
    new_device: pd.DataFrame,
    feature_order: list[str],
) -> tuple[pd.DataFrame, dict]:
    device_id = str(new_device.device_id.iloc[0])
    new_rows_before = len(new_device)
    excluded_new_job_rows = 0
    if device_id == "01-X450-1":
        excluded_new_job_rows = int(
            new_device.job_id.astype(str).isin(EXCLUDED_DEVICE01_JOBS).sum()
        )
    eligible_new_rows = new_rows_before - excluded_new_job_rows
    merged = old_device.merge(
        new_device,
        on=["device_id", "time_utc"],
        how="inner",
        validate="one_to_one",
        sort=True,
    )
    excluded_rows = 0
    if device_id == "01-X450-1":
        excluded = merged.job_id.astype(str).isin(EXCLUDED_DEVICE01_JOBS)
        excluded_rows = int(excluded.sum())
        merged = merged.loc[~excluded].copy()
    if merged.empty:
        raise RuntimeError(f"No merged rows remain for {device_id}")
    merged = merged.sort_values("time_utc", kind="stable").reset_index(drop=True)
    boundary = (
        merged.time_utc.diff().dt.total_seconds().ne(CADENCE_SECONDS)
        | merged.old_segment_id.ne(merged.old_segment_id.shift())
        | merged.new_chunk_id.ne(merged.new_chunk_id.shift())
        | merged.job_id.ne(merged.job_id.shift())
    )
    merged["segment_id"] = boundary.cumsum().astype("int32") - 1
    merged["target_observed_mask"] = (
        merged.old_observed_mask.astype(bool) & merged.new_observed_mask.astype(bool)
    )
    merged["time_beijing"] = pd.to_datetime(merged["time_beijing"])
    for feature in feature_order:
        merged[feature] = merged[feature].astype("float32")
    if merged[feature_order].isna().any().any():
        raise RuntimeError(f"Merged rows contain feature nulls for {device_id}")
    if not np.isfinite(merged[feature_order].to_numpy(dtype=np.float32, copy=False)).all():
        raise RuntimeError(f"Merged rows contain non-finite values for {device_id}")
    keep = [
        "time_utc",
        "time_beijing",
        "device_id",
        "job_id",
        "segment_id",
        "old_segment_id",
        "new_chunk_id",
        "old_observed_mask",
        "new_observed_mask",
        "target_observed_mask",
        "new_source_time_utc",
        "source_row",
        *feature_order,
    ]
    summary = {
        "device_id": device_id,
        "new_regular_rows": int(new_rows_before),
        "old_rows": int(len(old_device)),
        "merged_rows_before_bad_job_exclusion": int(len(merged) + excluded_rows),
        "new_rows_excluded_by_bad_job_policy": excluded_new_job_rows,
        "eligible_new_rows_after_bad_job_policy": eligible_new_rows,
        "excluded_bad_job_rows": excluded_rows,
        "merged_rows": int(len(merged)),
        "alignment_rate_vs_new_regular_rows": float((len(merged) + excluded_rows) / new_rows_before),
        "alignment_rate_after_bad_job_exclusion": float(len(merged) / eligible_new_rows),
        "jobs": int(merged.job_id.nunique()),
        "segments": int(merged.segment_id.nunique()),
    }
    return merged[keep], summary


def build_windows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    total_points = CONTEXT_POINTS + GAP_POINTS + PREDICTION_POINTS
    records: list[dict] = []
    candidates = 0
    excluded_imputed_target = 0
    target_observed = frame.target_observed_mask.to_numpy(dtype=bool)
    prefix = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(~target_observed)])
    for segment_id, group in frame.groupby("segment_id", sort=False):
        if len(group) < total_points:
            continue
        first = int(group.index[0])
        for local_start in range(0, len(group) - total_points + 1, WINDOW_STRIDE_POINTS):
            candidates += 1
            past_start = first + local_start
            past_end = past_start + CONTEXT_POINTS
            future_start = past_end + GAP_POINTS
            future_end = future_start + PREDICTION_POINTS
            if prefix[future_end] - prefix[future_start] != 0:
                excluded_imputed_target += 1
                continue
            records.append(
                {
                    "device_id": str(group.device_id.iloc[0]),
                    "job_id": str(group.job_id.iloc[0]),
                    "segment_id": int(segment_id),
                    "past_start_row": past_start,
                    "past_end_row_exclusive": past_end,
                    "gap_start_row": past_end,
                    "future_start_row": future_start,
                    "future_end_row_exclusive": future_end,
                    "past_end_time_utc": frame.at[past_end - 1, "time_utc"],
                    "future_start_time_utc": frame.at[future_start, "time_utc"],
                    "future_end_time_utc": frame.at[future_end - 1, "time_utc"],
                }
            )
    windows = pd.DataFrame(records)
    if windows.empty:
        raise RuntimeError(f"Device {frame.device_id.iloc[0]} has no usable windows")
    return windows, {
        "candidate_windows": candidates,
        "excluded_for_imputed_target": excluded_imputed_target,
        "usable_windows": int(len(windows)),
    }


def fit_scaler(device_dir: Path, devices: tuple[str, ...], features: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(len(features), dtype=np.float64)
    total_square = np.zeros(len(features), dtype=np.float64)
    count = 0
    for device in devices:
        values = pd.read_parquet(
            device_dir / device / "rows.parquet", columns=features
        ).to_numpy(dtype=np.float64, copy=False)
        total += values.sum(axis=0, dtype=np.float64)
        total_square += np.square(values).sum(axis=0, dtype=np.float64)
        count += len(values)
    if count == 0:
        raise RuntimeError("Cannot fit a scaler with zero rows")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return mean, scale, count


def validate_device_files(device_dir: Path, features: list[str]) -> None:
    for device in DEVICES:
        frame = pd.read_parquet(device_dir / device / "rows.parquet")
        windows = pd.read_parquet(device_dir / device / "windows.parquet")
        if set(frame.device_id.astype(str).unique()) != {device}:
            raise RuntimeError(f"Wrong device rows in {device}")
        if frame[features].isna().any().any():
            raise RuntimeError(f"Null feature values in {device}")
        if not (windows.past_end_row_exclusive - windows.past_start_row).eq(CONTEXT_POINTS).all():
            raise RuntimeError(f"Invalid context length in {device}")
        if not (windows.future_start_row - windows.past_end_row_exclusive).eq(GAP_POINTS).all():
            raise RuntimeError(f"Invalid latency gap in {device}")
        if not (windows.future_end_row_exclusive - windows.future_start_row).eq(PREDICTION_POINTS).all():
            raise RuntimeError(f"Invalid prediction length in {device}")
        observed = frame.target_observed_mask.to_numpy(dtype=bool)
        for row in windows.itertuples(index=False):
            if not observed[row.future_start_row : row.future_end_row_exclusive].all():
                raise RuntimeError(f"Imputed target survived window filtering in {device}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_root = args.old_root.resolve()
    new_data_dir = args.new_data_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    temporary = output.with_name(output.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"Remove or inspect stale temporary output: {temporary}")

    old_manifest_path = old_root / "feature_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_features = list(old_manifest["feature_order"])
    if len(old_features) != 128 or len(set(old_features)) != 128:
        raise ValueError("The source contract must contain exactly 128 unique old features")
    overlap = sorted(set(old_features) & set(ADDED_FEATURES))
    if overlap:
        raise ValueError(f"The six added features must be new names: {overlap}")
    feature_order = [*old_features, *ADDED_FEATURES]
    boolean_features = list(old_manifest["boolean_features"])
    continuous_features = [feature for feature in feature_order if feature not in boolean_features]

    try:
        temporary.mkdir(parents=True)
        device_dir = temporary / "devices"
        device_dir.mkdir()
        old = load_old_rows(old_root, old_features)
        new_chunk_summaries: list[dict] = []
        device_summaries: dict[str, dict] = {}
        next_chunk_id = 0
        for device in DEVICES:
            new_path = new_data_dir / f"influx_sensors_added_{device}_2026-07.parquet"
            if not new_path.is_file():
                raise FileNotFoundError(new_path)
            new, summaries, next_chunk_id = regularize_new_device(new_path, next_chunk_id)
            new_chunk_summaries.extend(summaries)
            merged, merge_summary = merge_device(
                old.loc[old.device_id.astype(str).eq(device)].copy(),
                new,
                feature_order,
            )
            merged = merged.reset_index(drop=True)
            windows, window_summary = build_windows(merged)
            target_dir = device_dir / device
            target_dir.mkdir()
            write_parquet(merged, target_dir / "rows.parquet")
            write_parquet(windows, target_dir / "windows.parquet")
            device_summaries[device] = {**merge_summary, **window_summary}
            print(
                f"{device}: rows={len(merged):,}, jobs={merged.job_id.nunique()}, "
                f"segments={merged.segment_id.nunique()}, windows={len(windows):,}",
                flush=True,
            )

        folds_dir = temporary / "folds"
        folds_dir.mkdir()
        fold_metadata = {}
        for name, split in FOLDS.items():
            fold_dir = folds_dir / name
            fold_dir.mkdir()
            mean, scale, rows = fit_scaler(device_dir, split["train"], feature_order)
            np.savez_compressed(
                fold_dir / "scaler.npz",
                mean=mean.astype(np.float32),
                scale=scale.astype(np.float32),
            )
            fold_metadata[name] = {
                "train_devices": list(split["train"]),
                "validation_devices": list(split["validation"]),
                "scaler_fit_rows": rows,
            }

        manifest = {
            "version": 1,
            "experiment": "D_latency_aware_old128_plus_new6_available134",
            "feature_order": feature_order,
            "old_features": old_features,
            "added_features": list(ADDED_FEATURES),
            "target_features": feature_order,
            "boolean_features": boolean_features,
            "continuous_features": continuous_features,
            "primary_objective_features": old_features,
            "auxiliary_output_features": list(ADDED_FEATURES),
            "primary_feature_indices": list(range(128)),
            "auxiliary_feature_indices": list(range(128, 134)),
            "cadence_seconds": CADENCE_SECONDS,
            "context_points": CONTEXT_POINTS,
            "context_seconds": CONTEXT_POINTS * CADENCE_SECONDS,
            "latency_gap_points": GAP_POINTS,
            "latency_gap_seconds": GAP_POINTS * CADENCE_SECONDS,
            "prediction_points": PREDICTION_POINTS,
            "prediction_seconds": PREDICTION_POINTS * CADENCE_SECONDS,
            "window_stride_points": WINDOW_STRIDE_POINTS,
            "window_stride_seconds": WINDOW_STRIDE_POINTS * CADENCE_SECONDS,
            "training_devices": list(TRAIN_DEVICES),
            "external_test_device": EXTERNAL_DEVICE,
            "excluded_device01_jobs": sorted(EXCLUDED_DEVICE01_JOBS),
            "same_name_policy": "The old 128 channels are authoritative; CSV duplicate names are not added.",
            "target_policy": "This data artifact exposes all 134 channels; the D experiment contract selects 129 dynamic history fields, five condition-only fields, and 27 outputs.",
            "folds": fold_metadata,
        }
        canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
        manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        (temporary / "feature_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_parquet(pd.DataFrame(new_chunk_summaries), temporary / "new_chunks.parquet")
        metadata = {
            "source_old_root": str(old_root),
            "source_new_data_dir": str(new_data_dir),
            "source_files_read_only": True,
            "device_statistics": device_summaries,
            "folds": fold_metadata,
            "new_time_grid_policy": "ceil source UTC time to the next global 2-second tick",
            "gap_policy": {
                "new_source_fill": "causal forward hold of at most one missing point",
                "hard_boundary_when_delta_seconds_gt": MAX_CONTIGUOUS_DELTA_SECONDS,
                "forecast_targets_require_old_and_new_observed": True,
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        validate_device_files(device_dir, feature_order)
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(f"DONE: {output}")


if __name__ == "__main__":
    main()

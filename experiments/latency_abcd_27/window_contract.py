"""Shared 2s cadence, 10s gap and strict window validation for A/B/C/D."""
import numpy as np
import pandas as pd

CONTEXT_POINTS = 150
GAP_POINTS = 5
PREDICTION_POINTS = 60
CADENCE_SECONDS = 2
INDEX_COLUMNS = ["past_start_row", "past_end_row_exclusive", "future_start_row", "future_end_row_exclusive"]


def validate_manifest(manifest):
    expected = dict(cadence_seconds=2, context_points=150, latency_gap_points=5,
                    latency_gap_seconds=10, prediction_points=60)
    if any(type(manifest.get(k)) is not int or manifest[k] != v for k, v in expected.items()):
        raise ValueError("Expected 300s history / 10s gap / 120s forecast. Rebuild data with rebuild_gap10.py; do not relabel old data.")


def validate_geometry(windows, row_count):
    if windows.empty or any(k not in windows for k in INDEX_COLUMNS):
        raise ValueError("Missing or empty window indices")
    if any(not np.issubdtype(windows[k].dtype, np.integer) for k in INDEX_COLUMNS):
        raise ValueError("Window indices must be integers")
    start, end, future, stop = (windows[k].to_numpy(dtype=np.int64) for k in INDEX_COLUMNS)
    if (np.any(start < 0) or np.any(stop > row_count) or
            np.any(end - start != CONTEXT_POINTS) or np.any(future - end != GAP_POINTS) or
            np.any(stop - future != PREDICTION_POINTS)):
        raise ValueError("Invalid window geometry: require 150 history + 5 gap + 60 labels within rows")
    return start, end, future, stop


def row_boundaries(rows, device):
    required = {"time_utc", "device_id", "job_id", "segment_id"}
    if required - set(rows):
        raise ValueError("Rows lack time/device/job/segment metadata")
    if rows[list(required)].isna().any().any() or set(rows.device_id.astype(str)) != {device}:
        raise ValueError("Invalid row identity")
    times = pd.to_datetime(rows.time_utc, utc=True, errors="raise")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Rows must have unique increasing UTC timestamps")
    boundary = times.diff().dt.total_seconds().ne(CADENCE_SECONDS)
    for field in ("device_id", "job_id", "segment_id"):
        boundary |= rows[field].ne(rows[field].shift())
    return boundary.to_numpy(dtype=bool)


def observed_rows(rows):
    field = "target_observed_mask" if "target_observed_mask" in rows else "observed_mask"
    if field not in rows or rows[field].isna().any() or not rows[field].isin([True, False, 0, 1]).all():
        raise ValueError("Rows require a valid observed target mask")
    return rows[field].to_numpy(dtype=bool)


def validate_windows(windows, rows, device):
    start, end, future, stop = validate_geometry(windows, len(rows))
    boundary = row_boundaries(rows, device)
    prefix = np.r_[0, np.cumsum(boundary)]
    if np.any(prefix[stop] != prefix[start + 1]):
        raise ValueError("Window crosses a task, segment or timestamp boundary")
    bad_labels = np.r_[0, np.cumsum(~observed_rows(rows))]
    if np.any(bad_labels[stop] != bad_labels[future]):
        raise ValueError("Window contains unobserved labels")
    for field in ("device_id", "job_id", "segment_id"):
        if field in windows and not np.array_equal(windows[field].astype(str).to_numpy(), rows[field].iloc[start].astype(str).to_numpy()):
            raise ValueError("Window identity differs from rows")


def build_windows(rows, device, stride=5):
    if type(stride) is not int or stride < 1:
        raise ValueError("Positive integer stride required")
    boundary = row_boundaries(rows, device)
    bad_labels = np.r_[0, np.cumsum(~observed_rows(rows))]
    cuts = np.r_[np.flatnonzero(boundary), len(rows)]
    records = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        for start in range(int(lo), int(hi) - CONTEXT_POINTS - GAP_POINTS - PREDICTION_POINTS + 1, stride):
            end = start + CONTEXT_POINTS
            future, stop = end + GAP_POINTS, end + GAP_POINTS + PREDICTION_POINTS
            if bad_labels[stop] != bad_labels[future]:
                continue
            record = dict(zip(INDEX_COLUMNS, (start, end, future, stop)))
            record.update({f: rows[f].iloc[start] for f in ("device_id", "job_id", "segment_id")})
            record.update(past_end_time_utc=rows.time_utc.iloc[end-1],
                          future_start_time_utc=rows.time_utc.iloc[future],
                          future_end_time_utc=rows.time_utc.iloc[stop-1])
            records.append(record)
    result = pd.DataFrame(records)
    validate_windows(result, rows, device)
    return result

"""Rebuild A/B/C/D indices into a new 10s-gap dataset; never train or alter sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from window_contract import build_windows, validate_manifest


def rebuild(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    code = Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError(output)
    if source in output.parents or output in source.parents or output == code or code in output.parents:
        raise ValueError("New output must not overlap source data or code")
    manifest = json.loads((source / "feature_manifest.json").read_text(encoding="utf-8"))
    devices = list(manifest["training_devices"]) + [manifest["external_test_device"]]
    # Rows and fold scalers are unchanged; only window geometry and its manifest change.
    stride = int(manifest.get("window_stride_points", 5))
    output.mkdir(parents=True)
    counts = {}
    for device in devices:
        src = source / "devices" / device / "rows.parquet"
        rows = pd.read_parquet(src)
        windows = build_windows(rows, device, stride)
        dest = output / "devices" / device
        dest.mkdir(parents=True)
        shutil.copy2(src, dest / "rows.parquet")
        windows.to_parquet(dest / "windows.parquet", index=False)
        counts[device] = len(windows)
        print(device, len(rows), "rows", len(windows), "10s-gap windows", flush=True)
    shutil.copytree(source / "folds", output / "folds")
    manifest.pop("manifest_sha256", None)
    manifest.update(context_points=150, context_seconds=300, latency_gap_points=5, latency_gap_seconds=10,
                    cadence_seconds=2, prediction_points=60, prediction_seconds=120,
                    window_stride_points=stride, window_stride_seconds=stride * 2,
                    source_manifest_sha256=hashlib.sha256((source / "feature_manifest.json").read_bytes()).hexdigest(),
                    rebuilt_window_counts=counts)
    validate_manifest(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    (output / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(dict(source=str(source), training_started=False,
        note="Rows and fold scalers copied unchanged; windows rebuilt with a 10s gap. Rerun HPO/confirmation/training; old weights are incompatible.",
        windows=counts), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.source, args.output)

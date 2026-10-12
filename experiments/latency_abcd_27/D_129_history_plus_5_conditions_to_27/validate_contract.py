from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Read-only validation; never builds a model or trains.")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((here / "experiment_spec.json").read_text(encoding="utf-8"))
    targets = json.loads((here / spec["target_contract"]).read_text(encoding="utf-8"))
    root = args.data_root.resolve()
    manifest = json.loads((root / "feature_manifest.json").read_text(encoding="utf-8"))
    source = list(manifest["feature_order"]); old = list(manifest["old_features"])
    dynamic = old if spec["dynamic_feature_policy"] == "old_features" else old + list(spec["dynamic_extra_features"])
    conditions = list(spec.get("condition_features", []))
    outputs = list(targets["continuous_features"]) + list(targets["boolean_features"])
    errors = []
    if len(source) != spec["expected_source_feature_count"]: errors.append("source field count")
    if len(dynamic) + len(conditions) != len(set(dynamic + conditions)): errors.append("duplicate inputs")
    if any(name not in source for name in dynamic + conditions): errors.append("missing inputs")
    if len(outputs) != 27 or any(name not in dynamic for name in outputs): errors.append("invalid targets")
    devices = [*manifest["training_devices"], manifest["external_test_device"]]
    device_rows = {}
    for device in devices:
        rows_path = root / "devices" / device / "rows.parquet"
        windows_path = root / "devices" / device / "windows.parquet"
        rows_file = pq.ParquetFile(rows_path)
        columns = set(rows_file.schema_arrow.names)
        if any(name not in columns for name in dynamic + conditions): errors.append(f"{device} parquet fields")
        windows_file = pq.ParquetFile(windows_path)
        windows = windows_file.read(columns=["past_start_row", "past_end_row_exclusive",
                                             "future_start_row", "future_end_row_exclusive"]).to_pydict()
        valid = (all(b - a == 150 for a, b in zip(windows["past_start_row"], windows["past_end_row_exclusive"]))
                 and all(a - b == 15 for a, b in zip(windows["future_start_row"], windows["past_end_row_exclusive"]))
                 and all(b - a == 60 for a, b in zip(windows["future_start_row"], windows["future_end_row_exclusive"])))
        if not valid: errors.append(f"{device} window geometry")
        device_rows[device] = {"rows": int(rows_file.metadata.num_rows),
                               "windows": int(windows_file.metadata.num_rows)}
    for fold in (*FOLDS, "final"):
        with np.load(root / "folds" / fold / "scaler.npz") as scaler:
            if scaler["mean"].shape != (len(source),) or scaler["scale"].shape != (len(source),):
                errors.append(f"{fold} scaler")
    report = {"status": "ok" if not errors else "failed", "experiment_id": spec["experiment_id"],
              "source_fields": len(source), "dynamic_history_fields": len(dynamic),
              "condition_fields": len(conditions), "total_input_fields": len(dynamic) + len(conditions),
              "output_fields": len(outputs), "device_rows": device_rows, "errors": errors,
              "training_started": False}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors: raise SystemExit(1)


FOLDS = ("fold_a", "fold_b", "fold_c")

if __name__ == "__main__":
    main()

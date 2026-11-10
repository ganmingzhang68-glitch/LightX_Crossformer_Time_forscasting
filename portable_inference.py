"""Inference runtime copied into model exports; independent of training repositories."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from forecast_windows import contiguous_segments


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class PortableModel:
    def __init__(self, bundle, device="auto"):
        self.root = Path(bundle).resolve()
        self.metadata = json.loads((self.root / "bundle.json").read_text(encoding="utf-8"))
        m = self.metadata
        if m.get("version") != 1 or m.get("experiment") not in ("matrix", "all90"):
            raise ValueError("Unsupported inference bundle")
        if m.get("bundle_sha") != digest({k: v for k, v in m.items() if k != "bundle_sha"}):
            raise ValueError("Bundle metadata checksum mismatch")
        weights_path = m.get("weights_path", "weights.pt")
        required_files = {weights_path, "predict.py", "forecast_windows.py", "runtime/__init__.py",
                          "runtime/model.py", "runtime/encoding.py", "runtime/crossformer.py"}
        if not required_files <= set(m["files"]):
            raise ValueError("Incomplete inference bundle")
        for relative, expected in m["files"].items():
            path = (self.root / relative).resolve()
            if self.root not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Bundle file checksum/path mismatch: " + relative)
        self.c, self.pre = m["storage_contract"], m["preprocessor"]
        self.g = self.c["groups"][m["group"]]
        if self.c["cadence_seconds"] != 2 or self.g["gap"] != 5 or m["forecast_gap_seconds"] != 10:
            raise ValueError("Inference requires a trained 10s gap / 2s cadence")
        if self.pre["contract_sha"] != digest(self.c):
            raise ValueError("Preprocessor and field contract mismatch")
        self.profile = m["model_contract"].get("profile") if m["experiment"] == "matrix" else None
        self.fields = self.c["targets"]
        self.selected = list(dict.fromkeys(self.profile["dynamic"] + self.profile["conditions"])) if self.profile else self.c["source_features"]
        self.quality_required = self.profile is None or self.profile["quality"] or self.profile["events"]
        self.masks = self.profile is None or self.profile["missing_masks"]
        self.indices = [self.c["source_features"].index(f) for f in self.fields]
        self.boolean_indices = [self.fields.index(f) for f in self.c["boolean"]]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu")
        # A distinct package per bundle prevents model/encoder collisions in one process.
        package = "_crossformer_bundle_" + digest([str(self.root), m["bundle_sha"]])[:20]
        if package not in sys.modules:
            spec = importlib.util.spec_from_file_location(package, self.root / "runtime/__init__.py",
                                                        submodule_search_locations=[str(self.root / "runtime")])
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            spec.loader.exec_module(module)
        encoder_type = importlib.import_module(package + ".encoding").Encoder
        model_type = importlib.import_module(package + ".model").ForecastModel
        self.encoder = encoder_type(self.c, self.pre, self.profile) if self.profile else encoder_type(self.c, self.pre)
        self.model = model_type(m["model_contract"], m["group"], self.pre, m["params"]).to(self.device)
        state = torch.load(self.root / weights_path, map_location="cpu", weights_only=True)
        # Historical releases retain the exact original best.pt bytes and checksum.
        # Validate its signed metadata before selecting the nested tensor state.
        if "model" in state:
            expected = dict(contract=m["model_contract"], group=m["group"], pre=self.pre,
                            params=m["params"], epoch=m["epoch"], code_sha=m["source_code_sha"],
                            signature=m["source_signature"])
            if any(state.get(k) != value for k, value in expected.items()):
                raise ValueError("Original checkpoint metadata differs from published bundle")
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def predict_batch(self, windows):
        encoded = []
        for raw, quality in windows:
            if raw.shape != (self.g["context"], len(self.c["source_features"])) or quality.shape != (self.g["context"], 3):
                raise ValueError("Wrong historical input shape")
            selected_indices = [self.c["source_features"].index(f) for f in self.selected]
            if np.isinf(raw).any() or not np.isfinite(raw[:, self.indices]).all() or not np.isfinite(quality).all():
                raise ValueError("Invalid historical values or quality channels")
            if not self.masks and not np.isfinite(raw[:, selected_indices]).all():
                raise ValueError("Selected input has missing values but missing_masks is disabled")
            encoded.append(self.encoder.encode(raw, quality))
        tensors = [torch.from_numpy(np.stack([item[i] for item in encoded])).to(self.device) for i in range(3)]
        with torch.inference_mode():
            result = self.model(*tensors).float().cpu().numpy()
        result = result * self.encoder.scale[self.indices] + self.encoder.mean[self.indices]
        if not np.isfinite(result).all():
            raise ValueError("Model produced nonfinite predictions")
        result[..., self.boolean_indices] = result[..., self.boolean_indices] >= 0.5
        return result


def predict_file(bundle, input_path, output_path, mode="latest", stride=5, batch_size=64, device="auto", include_target=False):
    if mode not in ("latest", "rolling") or stride < 1 or batch_size < 1:
        raise ValueError("Invalid prediction mode/stride/batch size")
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    engine = PortableModel(bundle, device)
    if engine.root == output_path or engine.root in output_path.parents or output_path in engine.root.parents:
        raise ValueError("Prediction output must not overwrite the model bundle")
    frame = pd.read_csv(input_path) if input_path.suffix.lower() == ".csv" else pd.read_parquet(input_path)
    time_column = next((f for f in ("time_utc", "time_beijing", "time") if f in frame), None)
    if time_column is None:
        raise ValueError("Aligned input requires time_utc/time_beijing/time")
    missing = set(engine.selected) - set(frame)
    if missing:
        raise ValueError("Missing selected fields: " + ", ".join(sorted(missing)))
    for field in engine.selected:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    if not np.isfinite(frame[engine.fields].to_numpy(dtype="float32")).all():
        raise ValueError("Original128 values must be finite after alignment")
    quality_fields = ["observed_mask", "new_observed", "new_age_seconds"]
    if engine.quality_required and set(quality_fields) - set(frame):
        raise ValueError("This model requires observed_mask, new_observed and new_age_seconds")
    for field in ("observed_mask", "new_observed"):
        if field in frame and (frame[field].isna().any() or not frame[field].isin([True, False, 0, 1]).all()):
            raise ValueError("Quality masks must contain boolean/0/1 values")
    if "new_age_seconds" in frame:
        ages = pd.to_numeric(frame.new_age_seconds, errors="raise").to_numpy()
        if not np.isfinite(ages).all() or (ages < 0).any():
            raise ValueError("Invalid new source age")
    times = pd.to_datetime(frame[time_column], utc=True)
    for field in ("available_time_utc", "source_time_utc"):
        if field in frame:
            source = pd.to_datetime(frame[field], utc=True)
            if source.isna().any() or (source > times).any():
                raise ValueError("Future or unknown source availability")
    windows = []
    for part, timestamps in contiguous_segments(frame, time_column):
        n, context, horizon = len(part), engine.g["context"], engine.g["horizon"]
        starts = [n-context] if mode == "latest" and n >= context else range(0, n-context-5-horizon+1, stride) if mode == "rolling" else []
        observed = part.observed_mask.to_numpy(dtype=bool) if "observed_mask" in part else np.ones(n, dtype=bool)
        raw = np.full((n, len(engine.c["source_features"])), np.nan, dtype="float32")
        for field in engine.selected:
            raw[:, engine.c["source_features"].index(field)] = part[field].to_numpy(dtype="float32")
        quality = np.column_stack([observed, part.new_observed if "new_observed" in part else np.ones(n),
            part.new_age_seconds.to_numpy(dtype="float32") / 300 if "new_age_seconds" in part else np.zeros(n)]).astype("float32")
        for start in starts:
            end, future = start + context, start + context + 5
            if mode == "rolling" and not observed[future:future+horizon].all():
                continue
            truth = raw[future:future+horizon, engine.indices] if include_target and future+horizon <= n else None
            identity = {k: part[k].iloc[end-1] for k in ("device_id", "job_id", "segment_id") if k in part}
            windows.append((raw[start:end], quality[start:end], timestamps.iloc[end-1], truth, identity))
    if mode == "latest":
        windows = windows[-1:]
    if not windows:
        raise ValueError("No eligible task-bounded contexts/observed labels")
    records = []
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset+batch_size]
        predicted = engine.predict_batch([(w[0], w[1]) for w in batch])
        for values, (_, _, origin, truth, identity) in zip(predicted, batch):
            for lead, row in enumerate(values):
                record = dict(identity, forecast_origin_time=origin,
                    prediction_time=origin+pd.Timedelta(seconds=2*(5+lead+1)), horizon_step=lead+1)
                record.update(zip(engine.fields, row.tolist()))
                if truth is not None:
                    record.update({"actual__"+f: float(v) for f, v in zip(engine.fields, truth[lead])})
                records.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    if output_path.suffix.lower() == ".csv":
        result.to_csv(output_path, index=False)
    else:
        result.to_parquet(output_path, index=False, compression="zstd")
    return dict(output=str(output_path), contexts=len(windows), forecast_rows=len(result), output_features=len(engine.fields),
                group=engine.metadata["group"], forecast_gap_seconds=10, bundle_sha=engine.metadata["bundle_sha"], device=str(engine.device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("latest", "rolling"), default="latest")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-target", action="store_true")
    args = parser.parse_args()
    print(json.dumps(predict_file(args.bundle, args.input, args.output, args.mode, args.stride,
        args.batch_size, args.device, args.include_target), ensure_ascii=False, indent=2))

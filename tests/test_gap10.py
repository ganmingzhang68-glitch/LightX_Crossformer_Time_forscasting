"""Regression tests for the 10s gap, legacy input rejection and task boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "experiments/latency_abcd_27"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEGACY))
from predict import contiguous_segments, make_windows, validate_prediction_contract
from window_contract import build_windows, validate_windows, validate_manifest, validate_geometry
from rebuild_gap10 import rebuild


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rows(n=240):
    return pd.DataFrame(dict(time_utc=pd.date_range("2026-07-01", periods=n, freq="2s", tz="UTC"),
        device_id="01-X450-1", job_id="a", segment_id=0, target_observed_mask=True, f=np.arange(n, dtype=np.float32)))


def manifest():
    return dict(cadence_seconds=2, context_points=150, latency_gap_points=5, latency_gap_seconds=10, prediction_points=60)


def test_prediction_exact_labels_and_gap_exclusion():
    frame = rows(160)
    expected = np.arange(155, 160, dtype=np.float32)[:, None]
    times = frame.time_utc
    sample = make_windows(frame, times, ["f"], 150, 5, 5, "rolling")
    assert len(sample) == 1
    assert sample[0][1] == times.iloc[149]
    np.testing.assert_array_equal(sample[0][2], expected)
    frame.loc[150:154, "f"] = 9999
    after = make_windows(frame, times, ["f"], 150, 5, 5, "rolling")[0]
    np.testing.assert_array_equal(sample[0][0], after[0])
    np.testing.assert_array_equal(after[2], expected)
    frame["observed_mask"] = True
    frame.loc[150:154, "observed_mask"] = False
    assert len(make_windows(frame, times, ["f"], 150, 5, 5, "rolling")) == 1
    frame.loc[155, "observed_mask"] = False
    assert make_windows(frame, times, ["f"], 150, 5, 5, "rolling") == []
    assert make_windows(frame.iloc[:159], times.iloc[:159], ["f"], 150, 5, 5, "rolling") == []


@pytest.mark.parametrize("identity", ["job_id", "segment_id", "device_id"])
def test_prediction_never_crosses_identity(identity):
    frame = rows(320)
    frame.loc[160:, identity] = 1 if identity == "segment_id" else "b"
    parts = list(contiguous_segments(frame, "time_utc"))
    assert [len(part) for part, _ in parts] == [160, 160]
    assert sum(len(make_windows(part, ts, ["f"], 150, 5, 5, "rolling")) for part, ts in parts) == 2
    frame = frame.iloc[:160].copy()
    frame.loc[80:, identity] = 1 if identity == "segment_id" else "b"
    assert all(not make_windows(part, ts, ["f"], 150, 5, 5, "rolling") for part, ts in contiguous_segments(frame, "time_utc"))


def test_prediction_duplicate_times_partitioned_before_dedup_and_latest_order():
    frame = rows(160)
    other = frame.copy()
    other["job_id"] = "b"
    parts = list(contiguous_segments(pd.concat([other, frame], ignore_index=True), "time_utc"))
    assert len(parts) == 2 and all(len(p) == 160 for p, _ in parts)
    future = frame.copy()
    future["time_utc"] += pd.Timedelta(days=1)
    future["job_id"] = "future"
    parts = list(contiguous_segments(pd.concat([future, frame], ignore_index=True), "time_utc"))
    assert parts[-1][0].job_id.iloc[0] == "future"


def test_legacy_prediction_contract_is_rejected_before_model_import(tmp_path):
    with pytest.raises(ValueError, match="trained with a 10s gap"):
        validate_prediction_contract(json.loads((ROOT/"artifacts/model_config.json").read_text()), tmp_path/"missing.pt")
    data = b"new exported test checkpoint"
    checkpoint = tmp_path/"new.pt"
    checkpoint.write_bytes(data)
    import hashlib
    config = dict(forecast_gap_points=5, forecast_gap_seconds=10, cadence_seconds=2,
                  checkpoint_sha256=hashlib.sha256(data).hexdigest())
    validate_prediction_contract(config, checkpoint)
    config["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_prediction_contract(config, checkpoint)


def test_legacy_window_validation_rejects_overlap_gap_bounds_and_boundaries():
    frame = rows()
    windows = build_windows(frame, "01-X450-1")
    first = windows.iloc[:1].copy()
    assert first.future_start_row.iloc[0] == 155
    assert (frame.time_utc.iloc[155] - frame.time_utc.iloc[149]).total_seconds() == 12
    for start in (100, 150, 165):
        bad = first.copy()
        bad["future_start_row"], bad["future_end_row_exclusive"] = start, start+60
        with pytest.raises(ValueError, match="geometry"):
            validate_windows(bad, frame, "01-X450-1")
    with pytest.raises(ValueError, match="geometry"):
        validate_geometry(first, 214)
    for field, value in (("job_id", "b"), ("segment_id", 1)):
        bad = frame.copy()
        bad.loc[160:, field] = value
        with pytest.raises(ValueError, match="boundary"):
            validate_windows(first, bad, "01-X450-1")
    bad = frame.copy()
    bad.loc[155, "target_observed_mask"] = False
    with pytest.raises(ValueError, match="unobserved"):
        validate_windows(first, bad, "01-X450-1")
    old = dict(manifest(), latency_gap_points=15, latency_gap_seconds=30)
    with pytest.raises(ValueError, match="Rebuild"):
        validate_manifest(old)


@pytest.mark.parametrize("folder", ["A_128_history_to_27", "C_129_history_plus_4_conditions_to_27", "D_129_history_plus_5_conditions_to_27", "B_133_history_to_27"])
def test_actual_legacy_loaders_enforce_gap_and_task_boundaries(tmp_path, folder):
    is_b = folder.startswith("B_")
    filename = "latency_fusion_common.py" if is_b else "experiment_common.py"
    mod = load("gap_test_"+folder, LEGACY/folder/filename)
    frame = rows()
    fields = [f"f{i}" for i in range(133)]
    frame = pd.concat([frame, pd.DataFrame(np.zeros((len(frame), 133), dtype=np.float32), columns=fields)], axis=1)
    dest = tmp_path/"devices/01-X450-1"
    dest.mkdir(parents=True)
    frame.to_parquet(dest/"rows.parquet", index=False)
    windows = build_windows(frame, "01-X450-1")
    windows.to_parquet(dest/"windows.parquet", index=False)
    m = dict(manifest(), training_devices=["01-X450-1"], external_test_device="04-X450-4",
        feature_order=fields, old_features=fields[:128], added_features=fields[128:], boolean_features=fields[:1])
    (tmp_path/"feature_manifest.json").write_text(json.dumps(m))
    def repository():
        return mod.ExperimentRepository(tmp_path if is_b else SimpleNamespace(manifest=m, data_root=tmp_path, source_features=fields))
    assert len(repository().windows["01-X450-1"]) > 0
    windows.loc[0, "future_start_row"], windows.loc[0, "future_end_row_exclusive"] = 100, 160
    windows.to_parquet(dest/"windows.parquet", index=False)
    with pytest.raises(ValueError, match="geometry"):
        repository()


def test_rebuild_gap10_writes_new_dataset_and_preserves_sources(tmp_path):
    source, output = tmp_path/"old", tmp_path/"new"
    dev = source/"devices/01-X450-1"
    dev.mkdir(parents=True)
    rows().to_parquet(dev/"rows.parquet", index=False)
    folds = source/"folds/fold_a"
    folds.mkdir(parents=True)
    (folds/"scaler.npz").write_bytes(b"unchanged scaler")
    old = dict(manifest(), latency_gap_points=15, latency_gap_seconds=30,
        training_devices=[], external_test_device="01-X450-1")
    original = json.dumps(old)
    (source/"feature_manifest.json").write_text(original)
    rebuild(source, output)
    validate_manifest(json.loads((output/"feature_manifest.json").read_text()))
    validate_windows(pd.read_parquet(output/"devices/01-X450-1/windows.parquet"), rows(), "01-X450-1")
    assert (source/"feature_manifest.json").read_text() == original
    assert (output/"folds/fold_a/scaler.npz").read_bytes() == b"unchanged scaler"
    with pytest.raises(FileExistsError):
        rebuild(source, output)


def test_d_data_builder_uses_ten_second_gap():
    mod = load("gap_test_d_prepare", LEGACY/"D_129_history_plus_5_conditions_to_27/prepare_latency_134_for_d.py")
    frame = rows()
    windows, _ = mod.build_windows(frame)
    assert mod.GAP_POINTS == 5
    validate_windows(windows, frame, "01-X450-1")
    assert windows.future_start_row.iloc[0] == 155


def test_prediction_main_emits_gap10_times_and_matching_truth(tmp_path, monkeypatch):
    # A deterministic test model exercises CLI timing/target output without a trained checkpoint.
    import hashlib
    import torch
    import predict
    class ToyModel:
        def __init__(self, config):
            self.config = config
        def load_state_dict(self, state, strict=True):
            return None
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, x):
            return torch.zeros((len(x), self.config.pred_len, self.config.enc_in), device=x.device)
    config = dict(seq_len=150, pred_len=5, enc_in=1, forecast_gap_points=5, forecast_gap_seconds=10,
                  cadence_seconds=2, boolean_threshold_physical=0.5)
    checkpoint = tmp_path/"new.pt"
    torch.save({}, checkpoint)
    config["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path/"config.json").write_text(json.dumps(config))
    (tmp_path/"manifest.json").write_text(json.dumps(dict(feature_order=["f"], boolean_features=[])))
    (tmp_path/"scaler.json").write_text(json.dumps(dict(feature_order=["f"], mean=[0], scale=[1])))
    frame = rows(160)
    frame.to_parquet(tmp_path/"rows.parquet", index=False)
    monkeypatch.setitem(sys.modules, "crossformer", SimpleNamespace(Crossformer=ToyModel,
        CrossformerConfig=SimpleNamespace(from_dict=lambda values: SimpleNamespace(**values))))
    monkeypatch.setattr(sys, "argv", ["predict.py", "--input", str(tmp_path/"rows.parquet"),
        "--output", str(tmp_path/"result.csv"), "--mode", "rolling", "--device", "cpu", "--include-target",
        "--config", str(tmp_path/"config.json"), "--checkpoint", str(checkpoint),
        "--manifest", str(tmp_path/"manifest.json"), "--scaler", str(tmp_path/"scaler.json")])
    predict.main()
    result = pd.read_csv(tmp_path/"result.csv")
    origin = pd.to_datetime(result.forecast_origin_time, utc=True)
    times = pd.to_datetime(result.prediction_time, utc=True)
    assert (times - origin).dt.total_seconds().tolist() == [12, 14, 16, 18, 20]
    assert result.actual__f.tolist() == [155, 156, 157, 158, 159]

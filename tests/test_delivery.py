import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.input_window_matrix_v1 import common as matrix_common
from experiments.all90_latency_v1 import common as all90_common
from experiments.input_window_matrix_v1.tests.test_matrix import synthetic_repo
from export_model import export_model
from portable_inference import PortableModel, predict_file

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


@pytest.mark.parametrize("common", [matrix_common, all90_common])
def test_code_identity_ignores_environments_outputs_tests_and_newlines(tmp_path, monkeypatch, common):
    (tmp_path/"model.py").write_bytes(b"# runtime\nx = 1\n")
    (tmp_path/"vendor").mkdir()
    (tmp_path/"vendor/backbone.py").write_bytes(b"# backbone\n")
    monkeypatch.setattr(common, "HERE", tmp_path)
    original = common.code_identity()
    for relative in [".venv/Lib/site-packages/module.py", "venv/site.py", "tests/test_new.py", "outputs/temporary.py", "__pycache__/cache.py"]:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# excluded fixture", encoding="utf-8")
    assert common.code_identity() == original
    (tmp_path/"model.py").write_bytes(b"# runtime\r\nx = 1\r\n")
    assert common.code_identity() == original
    (tmp_path/"model.py").write_bytes(b"# runtime\nx = 2\n")
    assert common.code_identity() != original


@pytest.mark.parametrize("folder", ["A_128_history_to_27", "B_133_history_to_27", "C_129_history_plus_4_conditions_to_27", "D_129_history_plus_5_conditions_to_27"])
def test_historical_models_forward_without_external_library(folder):
    import importlib.util
    legacy = ROOT/"experiments/latency_abcd_27"
    filename = "latency_fusion_common.py" if folder.startswith("B_") else "experiment_common.py"
    spec = importlib.util.spec_from_file_location("delivery_"+folder, legacy/folder/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    params = dict(d_model=16, n_heads=2, e_layers=1, d_ff=32, factor=2, dropout=0,
                  crossformer_seg_len=10, revin_affine=True)
    if folder.startswith("B_"):
        model = module.build_model(None, params)
        width = 133
    else:
        width = 128 if folder.startswith("A_") else 129
        model = module.build_crossformer(None, params, width)
    with torch.inference_mode():
        result = model(torch.zeros(1, 150, width), None, None, None)
    assert result.shape == (1, 60, width) and torch.isfinite(result).all()


def checkpoint_fixture(path, profile_name, group, experiment):
    from experiments.input_window_matrix_v1 import data as md, model as mm
    from experiments.all90_latency_v1 import data as ad, model as am
    common = matrix_common if experiment == "matrix" else all90_common
    c = common.contract()
    repo = synthetic_repo()
    repo.c = c
    data = md if experiment == "matrix" else ad
    pre = data.fit_preprocessor(repo, data.split_jobs(repo)["train"])
    params = dict(c["model_params"], d_model=16, n_heads=2, e_layers=1, d_ff=32, factor=2, condition_hidden=8)
    if experiment == "matrix":
        profiles, _ = common.selected_profiles([profile_name])
        profile = next(iter(profiles.values()))
        cc = common.runtime_contract(c, profile)
        model = mm.ForecastModel(cc, group, pre, params).eval()
    else:
        cc, profile = c, None
        model = am.ForecastModel(cc, group, pre, params).eval()
    spec = dict(contract_sha=common.digest(cc), preprocessor=pre, params=params, group=group,
                code_sha=common.code_identity())
    signature = common.digest(spec)
    path.mkdir()
    common.write_json(path/"spec.json", dict(spec, signature=signature))
    common.write_json(path/"result.json", dict(signature=signature))
    torch.save(dict(model=model.state_dict(), contract=cc, group=group, pre=pre, params=params, epoch=1,
                    seed=1, signature=signature, code_sha=common.code_identity()), path/"best.pt")
    job = next(iter(repo.values))
    return repo, job, model, pre, profile


@pytest.mark.parametrize("group", list(matrix_common.contract()["groups"]))
@pytest.mark.parametrize("profile,experiment", [("BASE128", "matrix"), ("ALL90", "matrix"), ("N90_U40_events", "matrix"), ("ALL90", "all90")])
def test_export_preserves_predictions_for_all_windows_and_encoders(tmp_path, group, profile, experiment):
    repo, job, original, pre, p = checkpoint_fixture(tmp_path/"trained", profile, group, experiment)
    exported = tmp_path/"bundle"
    report = export_model(tmp_path/"trained", exported, experiment)
    assert report["numerical_roundtrip"] is True and report["forecast_gap_seconds"] == 10
    portable = PortableModel(exported, "cpu")
    length = portable.g["context"]
    raw = repo.values[job][:length].copy()
    quality = np.ones((length, 3), dtype="float32")
    encoded = portable.encoder.encode(raw, quality)
    with torch.inference_mode():
        expected = original(*(torch.from_numpy(v.copy()).unsqueeze(0) for v in encoded)).numpy()
    expected = expected * portable.encoder.scale[portable.indices] + portable.encoder.mean[portable.indices]
    expected[..., portable.boolean_indices] = expected[..., portable.boolean_indices] >= 0.5
    np.testing.assert_allclose(portable.predict_batch([(raw, quality)]), expected, rtol=1e-5, atol=1e-5)
    (exported/"weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        PortableModel(exported, "cpu")


def test_export_inference_runs_outside_repository_and_checks_input(tmp_path):
    group = "G4_300_10_60"
    repo, job, _, _, _ = checkpoint_fixture(tmp_path/"trained", "ALL90", group, "matrix")
    export_model(tmp_path/"trained", tmp_path/"bundle")
    transferred = tmp_path/"transferred"
    shutil.copytree(tmp_path/"bundle", transferred)
    frame = pd.DataFrame(repo.values[job], columns=repo.c["source_features"])
    times = repo.rows[job].time_utc
    frame["time_utc"], frame["job_id"], frame["segment_id"], frame["device_id"] = times, "job", 0, repo.c["development_devices"][0]
    frame["observed_mask"], frame["new_observed"], frame["new_age_seconds"] = True, True, 0
    frame.to_parquet(tmp_path/"input.parquet", index=False)
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    args = [sys.executable, "-B", str(transferred/"predict.py"), "--input", str(tmp_path/"input.parquet"),
            "--output", str(tmp_path/"output.parquet"), "--mode", "rolling", "--include-target", "--device", "cpu"]
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    root_result = subprocess.run([sys.executable, "-B", str(ROOT/"predict.py"), "--bundle", str(transferred),
        "--input", str(tmp_path/"input.parquet"), "--output", str(tmp_path/"root_output.parquet"),
        "--mode", "rolling", "--include-target", "--device", "cpu"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert root_result.returncode == 0, root_result.stderr
    output = pd.read_parquet(tmp_path/"output.parquet")
    pd.testing.assert_frame_equal(output, pd.read_parquet(tmp_path/"root_output.parquet"))
    origins = pd.to_datetime(output.forecast_origin_time)
    lead = (pd.to_datetime(output.prediction_time) - origins).dt.total_seconds()
    assert lead.min() == 12 and lead.max() == 70
    assert len([f for f in repo.c["targets"] if f in output]) == 128
    assert output.forecast_origin_time.nunique() == 14
    first = output.iloc[0]
    expected = frame.loc[155, repo.c["targets"]].to_numpy(dtype=float)
    np.testing.assert_allclose(first[["actual__"+f for f in repo.c["targets"]]].to_numpy(dtype=float), expected)
    with pytest.raises(FileExistsError):
        predict_file(transferred, tmp_path/"input.parquet", tmp_path/"output.parquet")
    missing = frame.drop(columns="new_age_seconds")
    missing.to_parquet(tmp_path/"missing.parquet", index=False)
    with pytest.raises(ValueError, match="requires observed_mask"):
        predict_file(transferred, tmp_path/"missing.parquet", tmp_path/"invalid.parquet")


def test_export_rejects_unknown_experiment(tmp_path):
    with pytest.raises(ValueError, match="Unknown experiment"):
        export_model(tmp_path/"trained", tmp_path/"bundle", "unknown")


def test_portable_bundle_can_retain_exact_original_best_checkpoint(tmp_path):
    checkpoint_fixture(tmp_path/"trained", "ALL90", "G4_300_10_60", "matrix")
    export_model(tmp_path/"trained", tmp_path/"bundle")
    bundle = tmp_path/"bundle"
    shutil.copyfile(tmp_path/"trained/best.pt", bundle/"best.pt")
    metadata = json.loads((bundle/"bundle.json").read_text(encoding="utf-8"))
    metadata["files"].pop("weights.pt")
    metadata["weights_path"] = "best.pt"
    metadata["files"]["best.pt"] = hashlib.sha256((bundle/"best.pt").read_bytes()).hexdigest()
    metadata.pop("bundle_sha")
    metadata["bundle_sha"] = matrix_common.digest(metadata)
    (bundle/"bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
    model = PortableModel(bundle, "cpu")
    assert model.metadata["source_checkpoint_sha"] == metadata["files"]["best.pt"]
    metadata["source_signature"] = "wrong"
    metadata.pop("bundle_sha")
    metadata["bundle_sha"] = matrix_common.digest(metadata)
    (bundle/"bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint metadata"):
        PortableModel(bundle, "cpu")

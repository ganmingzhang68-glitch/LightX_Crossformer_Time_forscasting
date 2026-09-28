"""Verification only: no optimizer steps, fitting of model weights, or real 04 reads."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import contract, digest
from build_contract import build
from prepare import numeric_snapshot, regularize, fuse, common_windows
from data import Encoder, fit_preprocessor, Windows, split_jobs
from model import ForecastModel
from metrics import Accumulator
from compare import latency_stats, timing_summary, quality_summary
from benchmark import stratified_indices


@pytest.fixture
def c():
    return contract()


def raw_fixture(c, count=240):
    times = pd.date_range("2026-07-01", periods=count, freq="2s", tz="UTC")
    frame = pd.DataFrame({f: np.arange(count, dtype=float) + j for j, f in enumerate(c["added"])})
    frame["time_utc"], frame["source_row"] = times, np.arange(1, count + 1)
    for f in c["dates"]:
        frame[f] = "2026-06-30 08:00:00"
    for f in c["categorical"]:
        frame[f] = np.arange(count) % 2
    return frame


def repo_fixture(c):
    records, values, rows, origins = {}, {}, {}, {}
    for device in c["development_devices"]:
        for k in range(3):
            job = f"{device}_job{k}"
            n = 240
            x = np.tile(np.arange(218, dtype="float32"), (n, 1)) + np.arange(n)[:, None] * 0.01
            for f in c["boolean"]:
                x[:, c["targets"].index(f)] = (np.arange(n) // 7) % 2
            for f in c["categorical"]:
                x[:, c["source_features"].index(f)] = np.arange(n) % 2
            x[:, 128 + c["added"].index("平台到位")] = np.nan
            # First added setting is constant but must remain in the encoded model input.
            x[:, 129] = 450
            times = pd.date_range(f"2026-07-{k+1:02d}", periods=n, freq="2s", tz="UTC")
            meta = pd.DataFrame(dict(time_utc=times, segment_id=0, observed_mask=True,
                                     new_observed=True, new_age_seconds=0))
            records[job] = dict(device=device, start=times[0].isoformat())
            values[job], rows[job], origins[job] = x, meta, common_windows(meta, c)
    repo = SimpleNamespace(c=c, records=records, values=values, rows=rows, origins=origins, manifest={"external_only": False})
    repo.jobs = lambda dev: [j for j in records if records[j]["device"] == dev]
    return repo


def test_docs_are_actual_complete_source(c):
    root = Path(__file__).resolve().parents[3]
    assert c == build(root)
    assert len(c["source_features"]) == 218
    assert len(c["field_lineage"]) == 90
    assert all(r["included"] for r in c["field_lineage"])
    assert "平台到位" in c["added"] and len(c["dates"]) == 2
    assert c["groups"]["G4_300_10_60"] == dict(context=150, gap=5, horizon=30)


def test_dates_and_paired_results_cannot_look_forward(c):
    raw = raw_fixture(c, 4)
    date, result = c["dates"][0], next(iter(c["result_dates"]))
    raw.loc[1, date] = "2099-01-01 00:00:00"
    raw.loc[2, date] = "not a date"
    raw.loc[3, date] = None
    encoded = numeric_snapshot(raw, c)
    assert np.isclose(encoded.loc[0, date], np.log1p(1))
    assert encoded.loc[1:, [date, result]].isna().all().all()
    assert regularize(raw, c).loc[1, result] != regularize(raw, c).loc[1, result]  # real null never filled


def test_ceil_duplicate_short_hold_and_long_gap(c):
    raw = raw_fixture(c, 5)
    raw["time_utc"] = pd.to_datetime(["2026-07-01T00:00:01Z", "2026-07-01T00:00:02Z",
        "2026-07-01T00:00:06Z", "2026-07-01T00:00:12Z", "2026-07-01T00:00:14Z"])
    rows = regularize(raw, c)
    assert rows.iloc[0].source_row == 2
    assert rows.iloc[1].time_utc.second == 4 and not rows.iloc[1].new_observed
    assert rows.new_chunk.nunique() == 2
    assert (rows.available_time_utc <= rows.time_utc).all()
    delayed = regularize(raw, c, 3)
    assert (delayed.source_time_utc + pd.Timedelta(seconds=3) <= delayed.time_utc).all()


def test_all90_nulls_do_not_delete_whole_job(c):
    raw = raw_fixture(c)
    raw["平台到位"] = np.nan
    old = pd.DataFrame({f: np.ones(len(raw)) for f in c["targets"]})
    old["time_utc"], old["segment_id"], old["observed_mask"] = raw.time_utc, 0, True
    old.loc[10, "observed_mask"] = False
    old.loc[10, c["targets"][0]] = 999999
    rows = fuse(old, raw, "job", c)
    assert len(rows) == len(raw) and rows["平台到位"].isna().all()
    assert rows.loc[10, c["targets"][0]] == rows.loc[9, c["targets"][0]]


def test_shared_window_origin_exact_gap_labels_and_boundaries(c):
    repo = repo_fixture(c)
    split = split_jobs(repo)
    pre = fit_preprocessor(repo, split["train"])
    datasets = [Windows(repo, split["train"], group, pre) for group in c["groups"]]
    assert all(d.items == datasets[0].items for d in datasets)
    for ds in datasets:
        dynamic, condition, aux, target, _ = ds[0]
        assert dynamic.shape == (ds.g["context"], 130)
        assert condition.shape == (ds.g["context"], 90)
        assert aux.shape == (ds.g["context"], 93)
        assert target.shape == (ds.g["horizon"], 128)
        _, job, origin = ds.items[0]
        future = origin + ds.g["gap"] + 1
        assert (repo.rows[job].time_utc.iloc[future] - repo.rows[job].time_utc.iloc[origin]).total_seconds() == 12
        expected = repo.values[job][future, :128]
        np.testing.assert_allclose(target[0].numpy() * np.asarray(pre["scale"][:128]) + np.asarray(pre["mean"][:128]), expected, atol=1e-4)
    rows = repo.rows[job].copy()
    rows.loc[180, "observed_mask"] = False
    assert not len(common_windows(rows, c))  # every longest-window target spans this inserted point
    rows["observed_mask"] = True
    rows.loc[120:, "segment_id"] = 1
    assert not len(common_windows(rows, c))  # cannot cross hard segment boundary


def test_train_only_scaler_constants_masks_unknowns_and_all90(c):
    repo = repo_fixture(c)
    split = split_jobs(repo)
    pre = fit_preprocessor(repo, split["train"])
    for jobs in split["validation"].values():
        for job in jobs:
            repo.values[job][:] = 999999
    assert fit_preprocessor(repo, split["train"]) == pre
    with pytest.raises(ValueError, match="development"):
        fit_preprocessor(repo, {c["external_device"]: ["forbidden"]})
    encoder = Encoder(c, pre)
    raw = np.asarray(repo.values[next(iter(repo.records))][:5]).copy()
    f = c["categorical"][0]
    raw[0, c["source_features"].index(f)] = 9000
    dynamic, condition, aux = encoder.encode(raw, np.zeros((5, 3)))
    assert condition[0, c["added"].index(f)] == 1
    assert condition[0, 1] == 1  # retained 450 constant / 450 scale, not dropped
    platform = c["added"].index("平台到位")
    assert np.all(condition[:, platform] == 0) and np.all(aux[:, platform] == 0)
    # Every documented field owns an encoded value position and a mask position.
    assert condition.shape[1] == 90 and aux.shape[1] == 93 and np.isfinite(dynamic).all()


@pytest.mark.parametrize("group", ["G1_300_10_120", "G2_180_10_60", "G3_180_10_120", "G4_300_10_60"])
def test_forward_only_and_every_added_field_wired(c, group):
    torch.set_num_threads(1)
    repo = repo_fixture(c)
    split = split_jobs(repo)
    pre = fit_preprocessor(repo, split["train"])
    ds = Windows(repo, split["validation"], group, pre)
    params = dict(c["model_params"], d_model=16, n_heads=2, d_ff=32, e_layers=1, factor=1, condition_hidden=8)
    model = ForecastModel(c, group, pre, params).eval()
    dynamic, condition, aux, target, _ = ds[0]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        output = model(dynamic.unsqueeze(0), condition.unsqueeze(0), aux.unsqueeze(0))
    assert output.shape == (1, c["groups"][group]["horizon"], 128)
    assert torch.isfinite(output).all()
    assert sorted(model.scalar_positions + model.category_positions) == list(range(90))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_metrics_perfect_prediction_and_no_fake_zero_baseline(c):
    repo = repo_fixture(c)
    pre = fit_preprocessor(repo, split_jobs(repo)["train"])
    truth = np.zeros((2, 60, 128))
    truth[:, 10:40] = 1
    stats = Accumulator(c, pre)
    stats.add(truth, truth, np.zeros_like(truth))
    result = stats.finish()
    assert result["objective"] == 0 and result["boolean_macro_f1"] == 1
    zero = Accumulator(c, pre)
    zero.add(np.zeros_like(truth), np.zeros_like(truth), np.zeros_like(truth))
    assert zero.finish()["objective"] is None


def test_timing_pairs_quantiles_and_job_sampling():
    samples = [("04", "jobA", i) for i in range(30)] + [("04", "jobB", i) for i in range(5)]
    chosen = stratified_indices(samples, 8)
    assert len(chosen) == 8 and len({samples[i][1] for i in chosen}) == 2
    rows = [dict(model=f"{g}/{seed}", group=g, seed=seed, job_id=job, origin=i, round=0,
                 forward_ms=value, pipeline_ms=value*2)
            for g, value in [("G1", 4.0), ("G2", 2.0)] for seed in [1, 2]
            for job in ["A", "B"] for i in range(3)]
    result = timing_summary(pd.DataFrame(rows))
    assert result["paired_comparisons"][0]["metrics"]["pipeline_ms"]["median_paired_ratio"] == 0.5
    with pytest.raises(ValueError, match="Unpaired"):
        timing_summary(pd.DataFrame(rows[:-1]))
    assert latency_stats([1, 2, 3])["p50_ms"] == 2


def test_no_cpu_training_entrypoint(c, tmp_path, monkeypatch):
    from training import train
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        train(None, None, c, "G1_300_10_120", None, None, tmp_path / "no_run", 1, 1, 1)
    assert not (tmp_path / "no_run").exists()


def test_quality_comparison_common_horizon_and_unpaired_rejection():
    records = []
    for group, horizon, value in [("G1", 60, 0.3), ("G2", 30, 0.2), ("G3", 60, 0.4), ("G4", 30, 0.25)]:
        for seed in [1, 2, 3]:
            stat = dict(objective=value, continuous_relative_mse=value,
                        continuous_skill=1-value, boolean_macro_f1=1-value)
            records.append(dict(group=group, seed=seed, horizon_points=horizon, window_selection_sha="same",
                data_sha="same", active_targets=["f"], precision="fp32", **stat,
                common_first_60s=stat, by_job={j: stat for j in ["A", "B"]},
                common60_by_job={j: stat for j in ["A", "B"]}))
    summary = quality_summary(records)
    assert len(summary["paired_task_comparisons"]) == 8  # six common60 pairs plus two equal-horizon full pairs
    pair = summary["paired_task_comparisons"][0]
    assert np.isclose(pair["candidate_minus_baseline_J"]["mean"], -0.1)
    changed = copy.deepcopy(records)
    changed[0]["active_targets"] = ["different"]
    with pytest.raises(ValueError, match="Unpaired"):
        quality_summary(changed)


def test_selection_orchestration_freezes_all_groups_without_training(c, tmp_path, monkeypatch):
    import data
    import training
    import run
    from common import read_json
    repo = repo_fixture(c)
    monkeypatch.setattr(data, "Repository", lambda *args, **kwargs: repo)
    calls = []

    def fake_train(train_ds, val_ds, cc, group, pre, params, output, epochs, patience, seed, *args):
        calls.append((group, seed, train_ds.items, val_ds.items))
        assert all(item[0] != c["external_device"] for item in train_ds.items + val_ds.items)
        return dict(best_epoch=c["seeds"].index(seed)+2, best_score=0.5)

    monkeypatch.setattr(training, "train", fake_train)
    args = SimpleNamespace(contract=None, data_root=tmp_path / "data", output=tmp_path / "selection",
                           precision="fp32", resume=False, workers=0)
    run.select(args)
    frozen = read_json(args.output / "frozen.json")
    assert len(calls) == 12
    assert all(r["fixed_epochs"] == 3 for r in frozen["results"].values())
    assert all(call[2:] == calls[0][2:] for call in calls)
    run.verify_freeze(frozen, c, repo)
    assert not list(tmp_path.rglob("*.pt"))


def test_sharded_prepare_and_repository_roundtrip(c, tmp_path):
    from prepare import prepare
    from data import Repository
    from common import write_json
    old_root, added_root, output = tmp_path / "old", tmp_path / "added", tmp_path / "prepared"
    old_root.mkdir()
    added_root.mkdir()
    source = repo_fixture(c)
    all_old = []
    for dev in c["development_devices"]:
        added_parts, jobs = [], []
        for k, job in enumerate(source.jobs(dev)):
            times = source.rows[job].time_utc
            old = pd.DataFrame(source.values[job][:, :128], columns=c["targets"])
            old["time_utc"], old["device_id"], old["segment_id"], old["observed_mask"] = times, dev, k, True
            all_old.append(old)
            raw = raw_fixture(c, len(times))
            raw["time_utc"], raw["device_id"], raw["job_id"] = times, dev, job
            raw["平台到位"] = np.nan
            added_parts.append(raw)
            jobs.append(dict(device_id=dev, job_id=job, job_start_utc=times.iloc[0], job_end_utc=times.iloc[-1]))
        pd.concat(added_parts, ignore_index=True).to_parquet(added_root / f"influx_sensors_added_{dev}_2026-07.parquet", index=False)
        pd.DataFrame(jobs).to_parquet(added_root / f"influx_jobs_added_{dev}_2026-07.parquet", index=False)
    old = pd.concat(all_old, ignore_index=True)
    old.to_parquet(old_root / "train.parquet", index=False)
    for name in ("validation", "test"):
        old.iloc[:0].to_parquet(old_root / f"{name}.parquet", index=False)
    write_json(old_root / "feature_manifest.json", dict(feature_order=c["targets"], continuous_values_clipped=False))
    prepare(SimpleNamespace(contract=None, external=False, old_root=old_root, added_root=added_root,
                            output=output, stride=5, added_delay_seconds=0))
    repo = Repository(output, c)
    assert len(repo.records) == 9 and all(values.shape[1] == 218 for values in repo.values.values())
    split = split_jobs(repo)
    pre = fit_preprocessor(repo, split["train"])
    for group in c["groups"]:
        ds = Windows(repo, split["validation"], group, pre)
        assert len(ds) and all(torch.isfinite(v).all() for v in ds[0][:-1])
    # Corrupting a persisted matrix must be detected before training or evaluation.
    first = next(iter(repo.records))
    matrix_path = output / repo.records[first]["path"] / "values.npy"
    with matrix_path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"X")
    with pytest.raises(ValueError, match="Changed data file"):
        Repository(output, c)

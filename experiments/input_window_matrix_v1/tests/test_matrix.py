"""CPU forwards, synthetic pipeline and fake queue only: no optimizer updates."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import contract, catalog, digest, write_json, read_json, runtime_contract, selected_profiles, file_hash
from data import Encoder, Windows, split_jobs, fit_preprocessor, apply_shared_cohort, event_channels
from model import ForecastModel, grouped_loss
from metrics import evaluate
from training import loader
from train_queue import make_plan, execute_queue, frozen_pair, exclusive_lock, verify_plan

torch.set_num_threads(1)


def synthetic_repo():
    c = contract()
    records, values, rows, origins = {}, {}, {}, {}
    for device in c["development_devices"]:
        for k in range(3):
            job = f"{device}_task{k}"
            n = 250
            x = np.tile(np.arange(218, dtype="float32"), (n, 1)) + np.arange(n, dtype="float32")[:, None] * .01
            for f in c["boolean"]:
                x[:, c["source_features"].index(f)] = (np.arange(n)//7) % 2
            for f in c["categorical"]:
                x[:, c["source_features"].index(f)] = np.arange(n) % 2
            x[:, c["source_features"].index("平台到位")] = np.nan
            times = pd.date_range(f"2026-07-{k+1:02d}", periods=n, freq="2s", tz="UTC")
            meta = pd.DataFrame(dict(time_utc=times, segment_id=0, observed_mask=True, new_observed=True, new_age_seconds=0))
            from prepare import common_windows
            records[job] = dict(device=device, start=times[0].isoformat())
            values[job], rows[job], origins[job] = x, meta, common_windows(meta, c)
    repo = SimpleNamespace(c=c, records=records, values=values, rows=rows, origins=origins, manifest={"external_only": False})
    repo.jobs = lambda dev: [job for job in records if records[job]["device"] == dev and len(origins[job])]
    return repo


def test_inventory_default_output_matrix_and_dedup():
    cat = catalog()
    assert len(cat["profiles"]) == 27
    p = make_plan()
    assert len(p["profiles"]) == 22 and len(p["tasks"]) == 528
    assert p["aliases"] == {"A128_27": "BASE128"}
    assert all(v["targets"] == contract()["targets"] for v in p["profiles"].values())
    assert len([t for t in p["tasks"] if t["stage"] == "refit"]) == 264
    assert {t["group"] for t in p["tasks"]} == set(contract()["groups"])
    assert p["tasks"][263]["stage"] == "select" and p["tasks"][264]["stage"] == "refit"
    assert not cat["profiles"]["F133_133"]["default"]
    assert not cat["profiles"]["N90_C0_events"]["default"]
    # Opting in to old133 inputs also deduplicates the new128 outputs.
    extra = make_plan(["B133_27", "F133_133"])
    assert extra["aliases"] == {"F133_133": "B133_27"}


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"train_cap": -1}, {"seeds": [1, 1]}, {"profiles": ["unknown"]}, {"output_policy":"historical"}])
def test_bad_plans_fail(kwargs):
    with pytest.raises(ValueError):
        make_plan(**kwargs)


@pytest.mark.parametrize("group", list(contract()["groups"]))
def test_every_input_profile_forward_and_exact_targets(group):
    repo = synthetic_repo()
    # Include optional event encoders; all new configurations output original128.
    profiles, _ = selected_profiles(list(catalog()["profiles"]), output_policy="old128")
    cohort = apply_shared_cohort(repo, profiles)
    assert sum(cohort["windows_by_job"].values()) > 0
    split = split_jobs(repo)
    pre = fit_preprocessor(repo, split["train"])
    params = dict(repo.c["model_params"], d_model=16, n_heads=2, d_ff=32, e_layers=1, factor=2, condition_hidden=8)
    items = None
    for name, profile in profiles.items():
        ds = Windows(repo, split["train"], group, pre, profile)
        items = ds.items if items is None else items
        assert ds.items == items
        dynamic, conditions, aux, target, _ = ds[0]
        g = repo.c["groups"][group]
        assert tuple(target.shape) == (g["horizon"], 128)
        c = runtime_contract(repo.c, profile)
        model = ForecastModel(c, group, pre, params).eval()
        with torch.inference_mode():
            prediction = model(*(v.unsqueeze(0) for v in (dynamic, conditions, aux)))
            assert prediction.shape == target.unsqueeze(0).shape
            assert torch.isfinite(prediction).all() and torch.isfinite(grouped_loss(prediction, target.unsqueeze(0), c))
        _, job, origin = ds.items[0]
        future = origin + g["gap"] + 1
        assert (repo.rows[job].time_utc.iloc[future] - repo.rows[job].time_utc.iloc[origin]).total_seconds() == 12
        raw = target[0].numpy() * np.asarray(pre["scale"])[ds.target_indices] + np.asarray(pre["mean"])[ds.target_indices]
        np.testing.assert_allclose(raw, repo.values[job][future, ds.target_indices], atol=1e-4)


def test_default128_forward_and_metrics_primary_and_common27():
    repo = synthetic_repo()
    profiles, _ = selected_profiles(["D129_5_27", "ALL90"])
    apply_shared_cohort(repo, profiles)
    split = split_jobs(repo)
    pre = fit_preprocessor(repo, split["train"])
    params = dict(repo.c["model_params"], d_model=16, d_ff=32, e_layers=1, condition_hidden=8)
    for profile in profiles.values():
        c = runtime_contract(repo.c, profile)
        ds = Windows(repo, split["validation"], "G4_300_10_60", pre, profile)
        model = ForecastModel(c, "G4_300_10_60", pre, params)
        result = evaluate(model, loader(ds, batch_size=4), c, pre, torch.device("cpu"))
        assert len(result["by_feature"]) == 128
        assert len(result["shared27_common_first_60s"]["by_feature"]) == 27
        assert result["points"] == len(ds) * 30
        assert result["objective"] == result["common_first_60s"]["objective"]


def test_training_isolation_missing_masks_and_shared_cohort():
    repo = synthetic_repo()
    profiles, _ = selected_profiles(["ALL90", "N90_U40", "B133_27"])
    split = split_jobs(repo)
    pre = fit_preprocessor(repo, split["train"])
    for jobs in split["validation"].values():
        for job in jobs:
            repo.values[job][:] = 999
    assert fit_preprocessor(repo, split["train"]) == pre
    with pytest.raises(ValueError, match="development"):
        fit_preprocessor(repo, {repo.c["external_device"]: ["forbidden"]})
    encoder = Encoder(repo.c, pre, profiles["ALL90"])
    raw = repo.values[next(iter(repo.values))][:5].copy()
    raw[:, repo.c["source_features"].index("平台到位")] = np.nan
    d, c, aux = encoder.encode(raw, np.zeros((5,3)))
    assert aux.shape == (5,93) and c.shape == (5,90)
    assert (aux[:, repo.c["added"].index("平台到位")] == 0).all()
    repo = synthetic_repo()
    before = sum(map(len, repo.origins.values()))
    job = next(iter(repo.values))
    repo.values[job][100, repo.c["source_features"].index("设松粉S")] = np.nan
    shared = apply_shared_cohort(repo, profiles)
    assert sum(shared["windows_by_job"].values()) < before
    # All90's optional null doesn't delete the job; required selected history does.
    assert len(repo.origins[job]) == 0


def test_event_edges_do_not_cross_unknown_snapshots():
    raw = np.array([[0.],[1.],[1.],[0.],[0.]])
    observed = np.array([[True],[True],[False],[True],[True]])
    encoded = event_channels(raw, observed)
    assert encoded[0].sum() == 0
    assert encoded[1,0] == 1 and encoded[1,3] == 1
    assert encoded[2,3] == 0 and encoded[3,1] == 0 and encoded[4,3] == 0


def fake_queue(root, profiles=None):
    plan = make_plan(profiles or ["BASE128", "ALL90"], seeds=[1,2,3], epochs=4)
    plan.update(data_sha="synthetic", inputs_sha="synthetic")
    plan["plan_sha"] = digest(plan)
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", dict(state="ready", started_at=None, plan_sha=plan["plan_sha"],
        completed_tasks=0, total_tasks=len(plan["tasks"]), current_task=None,
        tasks={t["id"]: dict(state="pending") for t in plan["tasks"]}))
    return plan, dict(cohort={"origins_sha":"synthetic"})


def fake_task(output, plan, inputs, repo, task):
    dest = output / task["path"]
    dest.mkdir(parents=True, exist_ok=True)
    epoch = task["seed"] if task["stage"] == "select" else frozen_pair(output, plan, task)["fixed_epochs"]
    (dest / "best.pt").write_bytes(task["id"].encode())
    write_json(dest / "result.json", dict(best_epoch=epoch, signature=digest(task)))


def fake_verifier(output, plan, inputs, task):
    result = read_json(output / task["path"] / "result.json")
    return dict(signature=result["signature"], hashes={"best.pt":file_hash(output / task["path"] / "best.pt")},
        best_epoch=result["best_epoch"], parameter_count=1, training_seconds=1.)


def test_sequential_queue_failure_resume_and_complete_barrier(tmp_path):
    plan, inputs = fake_queue(tmp_path)
    calls, running = [], [0]
    def fail_once(*args):
        running[0] += 1
        assert running[0] == 1
        task = args[-1]
        calls.append(task["id"])
        try:
            if len(calls) == 4:
                raise RuntimeError("simulated interruption")
            fake_task(*args)
        finally:
            running[0] -= 1
    with pytest.raises(RuntimeError, match="simulated"):
        execute_queue(tmp_path, plan, inputs, None, task_runner=fail_once, verifier=fake_verifier)
    status = read_json(tmp_path / "status.json")
    assert status["state"] == "failed" and status["completed_tasks"] == 3
    assert not (tmp_path / "final.json").exists()
    assert all(t["state"] == "pending" for t in list(status["tasks"].values())[4:])
    with pytest.raises(ValueError, match="resume"):
        execute_queue(tmp_path, plan, inputs, None, task_runner=fake_task, verifier=fake_verifier)
    resumed = []
    def continue_task(*args):
        resumed.append(args[-1]["id"])
        fake_task(*args)
    execute_queue(tmp_path, plan, inputs, None, resume=True, task_runner=continue_task, verifier=fake_verifier)
    assert resumed[0] == plan["tasks"][3]["id"]
    assert len(resumed) == len(plan["tasks"]) - 3
    final = read_json(tmp_path / "final.json")
    assert len(final["models"]) == 24 and all(m["best_epoch"] == 2 for m in final["models"])
    assert read_json(tmp_path / "status.json")["state"] == "complete"
    # Changed finished weights are rejected, rather than silently considered done.
    (tmp_path / plan["tasks"][0]["path"] / "best.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        execute_queue(tmp_path, plan, inputs, None, resume=True, task_runner=fake_task, verifier=fake_verifier)
    assert read_json(tmp_path / "status.json")["state"] == "failed"


def test_plan_mutation_and_exclusive_lock(tmp_path):
    plan, _ = fake_queue(tmp_path, ["ALL90"])
    verify_plan(plan)
    changed = copy.deepcopy(plan)
    changed["selection"]["epochs"] += 1
    with pytest.raises(ValueError, match="Signature"):
        verify_plan(changed)
    with exclusive_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another"):
            with exclusive_lock(tmp_path):
                pass
    with exclusive_lock(tmp_path):
        pass


def test_real_preparation_init_and_external_completion_barrier(tmp_path):
    from train_queue import initialize
    from evaluate04 import completed_queue
    from data import Repository
    repo = synthetic_repo()
    source = tmp_path / "prepared"
    source.mkdir()
    manifest = dict(status="complete", contract_sha=digest(repo.c), external_only=False,
        devices=repo.c["development_devices"], stride=5, source_features=repo.c["source_features"], groups=repo.c["groups"], jobs=[],
        added_delay_seconds=0, cohort="shared synthetic origins", availability="synthetic observed snapshots", date_encoding="synthetic numeric ages", fill="synthetic no gaps")
    for job, rec in repo.records.items():
        rel = Path(rec["device"]) / job
        dest = source / rel
        dest.mkdir(parents=True)
        np.save(dest / "values.npy", repo.values[job])
        np.save(dest / "origins.npy", repo.origins[job])
        rows = repo.rows[job].copy()
        rows["available_time_utc"] = rows.time_utc
        rows.to_parquet(dest / "rows.parquet", index=False)
        manifest["jobs"].append(dict(**rec, job_id=job, path=rel.as_posix(), rows=len(rows),
            windows=len(repo.origins[job]), hashes={f:file_hash(dest/f) for f in ["values.npy","origins.npy","rows.parquet"]}))
    write_json(source / "manifest.json", manifest)
    output = tmp_path / "queue"
    args = SimpleNamespace(data_root=source, output=output, profiles=["BASE128","ALL90"],
        output_policy="old128", seeds=[1], epochs=2, patience=1, train_cap=0, workers=0)
    initialize(args)
    plan = read_json(output / "plan.json")
    verify_plan(plan)
    inputs = read_json(output / "inputs.json")
    assert digest(inputs) == plan["inputs_sha"]
    assert read_json(output / "preflight.json")["training_started"] is False
    assert read_json(output / "status.json")["state"] == "ready"
    loaded = Repository(source, repo.c)
    assert apply_shared_cohort(loaded, plan["profiles"]) == inputs["cohort"]
    # No final barrier document exists; evaluation cannot silently choose partial winners.
    with pytest.raises(ValueError, match="finish"):
        completed_queue(output)

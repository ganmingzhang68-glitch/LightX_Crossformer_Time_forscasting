"""Explicit quality/latency commands, only after the entire training queue is complete."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from common import contract, digest, read_json, write_json, safe_new_output
from train_queue import verify_plan, artifact_record
from data import Repository, Windows, apply_shared_cohort
from training import load_model, loader
from metrics import evaluate


def completed_queue(output):
    plan = read_json(output / "plan.json")
    verify_plan(plan)
    inputs, status = (read_json(output / f) for f in ("inputs.json", "status.json"))
    if status["state"] != "complete":
        raise ValueError("All input/window/seed models must finish before external evaluation")
    final = read_json(output / "final.json")
    if digest(inputs) != plan["inputs_sha"] or final["plan_sha"] != plan["plan_sha"] or status["plan_sha"] != plan["plan_sha"]:
        raise ValueError("Queue identity mismatch")
    if final["data_sha"] != plan["data_sha"] or final["code_sha"] != plan["code_sha"] or final["cohort"] != inputs["cohort"]:
        raise ValueError("Final matrix source/cohort identity differs")
    expected = {(t["profile"], t["group"], t["seed"]) for t in plan["tasks"] if t["stage"] == "refit"}
    if (status["state"] != "complete" or final["status"] != "complete" or
        status["completed_tasks"] != len(plan["tasks"]) or len(final["models"]) != len(expected) or
        {(r["profile"], r["group"], r["seed"]) for r in final["models"]} != expected):
        raise ValueError("All input/window/seed models must finish before external evaluation")
    lookup = {(r["profile"], r["group"], r["seed"]): r for r in final["models"]}
    for task in plan["tasks"]:
        if status["tasks"][task["id"]]["state"] != "complete":
            raise ValueError("Incomplete task")
        actual = artifact_record(output, plan, inputs, task)
        if actual != status["tasks"][task["id"]]["artifact"]:
            raise ValueError("A completed training artifact changed")
        if task["stage"] == "refit":
            record = lookup[task["profile"], task["group"], task["seed"]]
            dest = (output / record["path"]).resolve()
            if (record["path"] != task["path"] or output.resolve() not in dest.parents or
                record["checkpoint_sha"] != actual["hashes"]["best.pt"] or
                any(record[k] != actual[k] for k in ("best_epoch", "parameter_count", "training_seconds"))):
                raise ValueError("Final checkpoint path/manifest differs")
    return plan, inputs, final


def external_context(args):
    plan, inputs, final = completed_queue(args.model_root)
    repo = Repository(args.data_root, contract(), external=True)
    if {k: repo.manifest[k] for k in plan["data_policy"]} != plan["data_policy"]:
        raise ValueError("External cadence/fill/source-delay policy differs from development")
    cohort = apply_shared_cohort(repo, plan["profiles"])
    if any(cohort[k] != inputs["cohort"][k] for k in ("required_history_fields", "observed_added_targets")):
        raise ValueError("External cohort policy differs from development")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return plan, inputs, final, repo, cohort, device


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quality(args):
    plan, inputs, final, repo, cohort, device = external_context(args)
    out = safe_new_output(args.output, [args.model_root, args.data_root])
    out.mkdir(parents=True)
    selection = {repo.c["external_device"]: repo.jobs(repo.c["external_device"])}
    summary = []
    for record in final["models"]:
        p = plan["profiles"][record["profile"]]
        model, state, _ = load_model(args.model_root / record["path"], device)
        ds = Windows(repo, selection, record["group"], state["pre"], p)
        metrics = evaluate(model, loader(ds, args.batch_size), state["contract"], state["pre"], device)
        metrics.update(profile=record["profile"], group=record["group"], seed=record["seed"], precision="fp32",
            output_fields=len(p["targets"]), checkpoint_sha=record["checkpoint_sha"],
            window_selection_sha=digest(ds.items), data_sha=digest(repo.manifest), external_exploratory=True)
        path = Path(record["profile"]) / record["group"] / f"seed_{record['seed']}.json"
        write_json(out / path, metrics)
        summary.append(dict(profile=record["profile"], group=record["group"], seed=record["seed"], path=path.as_posix(),
            full_J=metrics["objective"], common60_J=metrics["common_first_60s"]["objective"],
            shared27_common60_J=metrics["shared27_common_first_60s"]["objective"]))
        del model, ds
        release()
        print("Evaluated", path, flush=True)
    write_json(out / "evaluation.json", dict(status="complete", plan_sha=plan["plan_sha"], cohort=cohort,
        data_sha=digest(repo.manifest), records=summary,
        note="04 exploratory assessment. Use identical target sets, shared origins and common60 labels for cross-model comparisons."))


def sample_indices(items, count):
    jobs = {}
    for i, (_, job, _) in enumerate(items):
        jobs.setdefault(job, []).append(i)
    if count < len(jobs):
        raise ValueError("Samples must cover every eligible task")
    quotas = dict.fromkeys(jobs, 0)
    remaining = min(count, len(items))
    while remaining:
        for job, indices in jobs.items():
            if remaining and quotas[job] < len(indices):
                quotas[job] += 1
                remaining -= 1
    return [indices[i] for job, indices in jobs.items() for i in np.linspace(0, len(indices)-1, quotas[job], dtype=int)]


def timed(function, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tick = time.perf_counter_ns()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter_ns()-tick)/1e6, result


def benchmark(args):
    import subprocess
    import os
    if min(args.samples, args.rounds, args.warmup) < 1:
        raise ValueError("Positive timing budgets required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan, inputs, final, repo, cohort, device = external_context(args)
    processes = []
    if device.type == "cuda":
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True)
        processes = [line for line in r.stdout.splitlines() if line.strip() and line.split(",")[0].strip() != str(os.getpid())]
        if processes:
            raise RuntimeError("Other GPU processes detected; timing requires an idle GPU")
    out = safe_new_output(args.output, [args.model_root, args.data_root])
    out.mkdir(parents=True)
    selection = {repo.c["external_device"]: repo.jobs(repo.c["external_device"])}
    reference = Windows(repo, selection, final["models"][0]["group"], inputs["refit_pre"], plan["profiles"][final["models"][0]["profile"]])
    indices = sample_indices(reference.items, args.samples)
    rng, rows = np.random.default_rng(20260928), []
    metadata = dict(status="running", plan_sha=plan["plan_sha"], data_sha=digest(repo.manifest), cohort=cohort,
        cohort_sha=digest([reference.items[i] for i in indices]), device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        torch=torch.__version__, cuda=torch.version.cuda, precision="fp32", batch_size=1, threads=1,
        warmup=args.warmup, samples=len(indices), rounds=args.rounds,
        scheduling="one model resident at a time; randomized model blocks each round, randomized requests and timing scopes within each block",
        pipeline_scope="aligned/type-decoded in-memory fields -> encoding -> transfer -> forward -> CPU inverse scaling/boolean threshold",
        excluded="disk/network/acquisition/fusion/date parsing/checkpoint loading/nominal 10s gap",
        source_processes=processes,
        note="GPU server timing on device04 data, not device04 deployment hardware. Model-block order can retain thermal/time drift.")
    write_json(out / "benchmark.json", metadata)
    with torch.inference_mode():
        for round_id in range(args.rounds):
            for which in rng.permutation(len(final["models"])):
                record = final["models"][int(which)]
                p = plan["profiles"][record["profile"]]
                model, state, _ = load_model(args.model_root / record["path"], device)
                ds = Windows(repo, selection, record["group"], state["pre"], p)
                if ds.items != reference.items:
                    raise ValueError("Timing models don't share exactly the same origins")
                ti = ds.target_indices
                def pipeline(raw, quality):
                    encoded = ds.encoder.encode(raw, quality)
                    output = model(*(torch.from_numpy(v.copy()).unsqueeze(0).to(device) for v in encoded)).float().cpu().numpy()
                    output = output * ds.encoder.scale[ti] + ds.encoder.mean[ti]
                    bi = [p["targets"].index(f) for f in p["boolean"]]
                    output[..., bi] = output[..., bi] >= 0.5
                    if not np.isfinite(output).all():
                        raise ValueError("Nonfinite inference")
                    return output
                for k in range(args.warmup):
                    raw, qual = ds.raw_input(indices[k % len(indices)])
                    pipeline(raw.copy(), qual.copy())
                for index in rng.permutation(indices):
                    raw, qual = (v.copy() for v in ds.raw_input(int(index)))
                    encoded = [torch.from_numpy(v.copy()).unsqueeze(0).to(device) for v in ds.encoder.encode(raw, qual)]
                    operations = {"forward_ms": lambda: model(*encoded), "pipeline_ms": lambda: pipeline(raw, qual)}
                    measurements = {}
                    for scope in rng.permutation(list(operations)):
                        measurements[str(scope)], value = timed(operations[str(scope)], device)
                        if tuple(value.shape) != (1, ds.g["horizon"], len(p["targets"])):
                            raise ValueError("Wrong timing output shape")
                    rows.append(dict(profile=record["profile"], group=record["group"], seed=record["seed"],
                        job_id=ds.items[int(index)][1], origin=ds.items[int(index)][2], round=round_id, **measurements))
                del model, ds, encoded, value
                release()
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "latency_samples.csv", index=False)
    summaries = []
    for (profile, group), part in frame.groupby(["profile", "group"], sort=False):
        summaries.append(dict(profile=profile, group=group, requests=len(part),
            **{f"{scope}_p{q}": float(np.percentile(part[scope], q)) for scope in ("forward_ms", "pipeline_ms") for q in (50,95,99)}))
    write_json(out / "timing_summary.json", summaries)
    metadata["status"] = "complete"
    write_json(out / "benchmark.json", metadata)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["quality", "benchmark"])
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=50)
    args = p.parse_args()
    (quality if args.command == "quality" else benchmark)(args)

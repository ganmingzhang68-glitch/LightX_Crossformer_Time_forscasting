"""Paired device04 batch-1 latency for already trained, frozen models."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import torch

from common import contract, digest, safe_new_output, write_json
from data import Repository, Windows
from run import final_models
from compare import timing_summary


def stratified_indices(items, count):
    """Spread requested inputs over jobs and each job's entire time range."""
    by_job = {}
    for i, (_, job, _) in enumerate(items):
        by_job.setdefault(job, []).append(i)
    if count < len(by_job):
        raise ValueError("Timing sample count must cover every eligible job")
    quotas = {job: 0 for job in by_job}
    remaining = min(count, len(items))
    while remaining:
        for job, indices in by_job.items():
            if quotas[job] < len(indices) and remaining:
                quotas[job] += 1
                remaining -= 1
    return [indices[i] for job, indices in by_job.items()
            for i in np.linspace(0, len(indices)-1, quotas[job], dtype=np.int64)]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(function, device):
    synchronize(device)
    start = time.perf_counter_ns()
    value = function()
    synchronize(device)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return elapsed, value


def forward(model, encoded, precision):
    with torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else nullcontext():
        return model(*encoded)


def pipeline(model, ds, raw, quality, device, precision):
    encoded = ds.encoder.encode(raw, quality)
    tensors = [torch.from_numpy(v.copy()).unsqueeze(0).to(device) for v in encoded]
    result = forward(model, tensors, precision).float().cpu().numpy()
    result = result * ds.encoder.scale[:128] + ds.encoder.mean[:128]
    boolean = [ds.c["targets"].index(f) for f in ds.c["boolean"]]
    result[..., boolean] = result[..., boolean] >= 0.5
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite inference result")
    return result


def environment(device):
    result = dict(platform=platform.platform(), cpu=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=str(device),
        intraop_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        tf32=False, compilation="eager", batch_size=1, timer="perf_counter_ns; CUDA synchronized before/after each request")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        result.update(gpu=prop.name, gpu_memory_bytes=prop.total_memory, gpu_capability=list(torch.cuda.get_device_capability(device)))
    return result


def gpu_processes():
    try:
        command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        return [row.strip() for row in result.stdout.splitlines() if row.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def benchmark(args):
    if args.samples < 1 or args.rounds < 1 or args.warmup < 1 or args.threads < 1:
        raise ValueError("Positive sample/round/warmup/thread counts required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 requires CUDA")
    processes = gpu_processes() if device.type == "cuda" else []
    others = [p for p in processes or [] if p.split(",")[0].strip() != str(os.getpid())]
    if others and not args.allow_busy_device:
        raise RuntimeError("Other GPU compute processes detected. Run timing on an idle GPU, or explicitly record contention with --allow-busy-device")
    c = contract(args.contract)
    repo = Repository(args.data_root, c, external=True)
    final, models = final_models(args.model_root, c, repo, device)
    out = safe_new_output(args.output, [args.data_root, args.model_root])
    out.mkdir(parents=True)
    selection = {c["external_device"]: repo.jobs(c["external_device"])}
    entries = []
    for record, model, state in models:
        ds = Windows(repo, selection, record["group"], state["pre"])
        entries.append(dict(record=record, model=model, dataset=ds, name=f"{record['group']}/seed_{record['seed']}"))
    reference = entries[0]["dataset"]
    if any(e["dataset"].items != reference.items for e in entries):
        raise ValueError("Timing requires exactly the same forecast origins")
    selected = stratified_indices(reference.items, args.samples)
    metadata = dict(status="running", environment=environment(device), precision=args.precision,
        data_sha=digest(repo.manifest), freeze_sha=final["frozen"]["freeze_sha"],
        models=[e["record"] for e in entries], warmup_per_model=args.warmup, rounds=args.rounds,
        requested_samples=args.samples, actual_samples=len(selected), sampling_seed=args.seed,
        cohort_sha=digest([reference.items[i] for i in selected]),
        source_processes=processes, contention_detected=bool(others),
        gpu_contention_check="unavailable" if processes is None else "checked before measurement",
        forward_scope="Encoded tensors already resident on target device; synchronized model forward wall time",
        pipeline_scope="Aligned/type-decoded in-memory 218 fields -> scale/category/masks -> tensor+device transfer -> model -> CPU inverse scaling+boolean threshold",
        excluded="Disk/network/acquisition/fusion/date parsing/checkpoint load and nominal 10s gap; no actual source-arrival timestamps",
        model_loading="All compared models resident; eager batch1 requests interleaved, no concurrent model execution")
    write_json(out / "benchmark.json", metadata)
    records = []
    rng = np.random.default_rng(args.seed)
    with torch.inference_mode():
        for entry in entries:
            ds, model = entry["dataset"], entry["model"]
            for k in range(args.warmup):
                raw, quality = ds.raw_input(selected[k % len(selected)])
                encoded = [torch.from_numpy(v.copy()).unsqueeze(0).to(device) for v in ds.encoder.encode(raw, quality)]
                forward(model, encoded, args.precision)
                pipeline(model, ds, raw, quality, device, args.precision)
            synchronize(device)
        for round_id in range(args.rounds):
            for index in rng.permutation(selected):
                for which in rng.permutation(len(entries)):
                    entry = entries[which]
                    ds, model = entry["dataset"], entry["model"]
                    raw, quality = ds.raw_input(int(index))  # exclude mmap/metadata lookup
                    raw, quality = raw.copy(), quality.copy()  # page faults outside timed scope
                    encoded = [torch.from_numpy(v.copy()).unsqueeze(0).to(device) for v in ds.encoder.encode(raw, quality)]
                    operations = dict(forward_ms=lambda: forward(model, encoded, args.precision),
                                      pipeline_ms=lambda: pipeline(model, ds, raw, quality, device, args.precision))
                    measured = {}
                    for scope in rng.permutation(["forward_ms", "pipeline_ms"]):
                        measured[str(scope)], result = timed(operations[str(scope)], device)
                        if tuple(result.shape) != (1, ds.g["horizon"], 128):
                            raise ValueError("Invalid benchmark output")
                    dev, job, origin = ds.items[index]
                    records.append(dict(model=entry["name"], group=entry["record"]["group"], seed=entry["record"]["seed"],
                                        job_id=job, origin=origin, round=round_id, **measured))
            pd.DataFrame(records).to_csv(out / "latency_samples.csv", index=False)
            print("Timing round", round_id + 1, "of", args.rounds, "complete", flush=True)
    metadata["status"] = "complete"
    metadata["ending_gpu_processes"] = gpu_processes() if device.type == "cuda" else []
    write_json(out / "timing_summary.json", timing_summary(pd.DataFrame(records)))
    write_json(out / "benchmark.json", metadata)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contract", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--allow-busy-device", action="store_true")
    benchmark(p.parse_args())

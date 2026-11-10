"""Paired whole-task uncertainty; never selects epochs or weights using device04."""
from __future__ import annotations

import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from common import read_json, write_json


def bootstrap_mean(values, repetitions=2000, seed=20260928):
    x = np.asarray(values, dtype="float64")
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Bootstrap requires finite task summaries")
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(repetitions)])
    return dict(mean=float(x.mean()), ci95=[float(v) for v in np.quantile(means, [0.025, 0.975])], tasks=len(x))


def quality_summary(records):
    if not records:
        raise ValueError("No evaluation records")
    for key in ("window_selection_sha", "data_sha", "active_targets", "precision"):
        if any(r[key] != records[0][key] for r in records):
            raise ValueError(f"Unpaired quality comparison: {key}")
    groups = sorted({r["group"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    lookup = {(r["group"], r["seed"]): r for r in records}
    if len(lookup) != len(records) or set(lookup) != set(itertools.product(groups, seeds)):
        raise ValueError("All groups require the same complete seed set")
    summaries = {}
    keys = ["objective", "continuous_relative_mse", "continuous_skill", "boolean_macro_f1"]
    for group in groups:
        summaries[group] = {}
        for scope in ("full", "common_first_60s"):
            stats = [lookup[group, s] if scope == "full" else lookup[group, s][scope] for s in seeds]
            summaries[group][scope] = {k: dict(mean=float(np.mean([r[k] for r in stats])),
                std=float(np.std([r[k] for r in stats], ddof=1)) if len(seeds) > 1 else 0.0)
                for k in keys if all(r[k] is not None for r in stats)}
    pairs = []
    for a, b in itertools.combinations(groups, 2):
        scopes = ["common60_by_job"]
        if lookup[a, seeds[0]]["horizon_points"] == lookup[b, seeds[0]]["horizon_points"]:
            scopes.append("by_job")
        for scope in scopes:
            jobs = set(lookup[a, seeds[0]][scope])
            if any(set(lookup[g, s][scope]) != jobs for g in (a, b) for s in seeds):
                raise ValueError("Task sets differ")
            values, excluded = [], []
            for job in sorted(jobs):
                diffs = []
                for seed in seeds:
                    av, bv = lookup[a, seed][scope][job]["objective"], lookup[b, seed][scope][job]["objective"]
                    if av is None or bv is None:
                        break
                    diffs.append(bv - av)
                if len(diffs) == len(seeds):
                    values.append(float(np.mean(diffs)))
                else:
                    excluded.append(job)
            pairs.append(dict(baseline=a, candidate=b, scope=scope, candidate_minus_baseline_J=
                bootstrap_mean(values) if values else None, excluded_unscorable_tasks=excluded))
    return dict(groups=summaries, paired_task_comparisons=pairs, seeds=seeds,
        note="Full metrics use each group's horizon; common60 uses exactly the same origins and +12..70s labels. Bootstrap averages paired seeds within each task, then resamples whole tasks. It describes this device's tasks, not unseen-device uncertainty.")


def latency_stats(values):
    x = np.asarray(values, dtype="float64")
    if not len(x) or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError("Latencies must be positive and finite")
    out = dict(n=len(x), mean_ms=float(x.mean()), std_ms=float(x.std()), min_ms=float(x.min()), max_ms=float(x.max()))
    out.update({f"p{p}_ms": float(np.percentile(x, p)) for p in (50, 90, 95, 99)})
    return out


def timing_summary(frame):
    models, groups = {}, {}
    for name, part in frame.groupby("model"):
        models[name] = {col: latency_stats(part[col]) for col in ("forward_ms", "pipeline_ms")}
    for group, part in frame.groupby("group"):
        groups[group] = {col: latency_stats(part[col]) for col in ("forward_ms", "pipeline_ms")}
    pairs = []
    for a, b in itertools.combinations(sorted(frame.group.unique()), 2):
        left, right = frame[frame.group == a], frame[frame.group == b]
        keys = ["seed", "job_id", "origin", "round"]
        pair = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        if len(pair) != len(left) or len(pair) != len(right):
            raise ValueError("Unpaired timing samples")
        metrics = {}
        for col in ("forward_ms", "pipeline_ms"):
            delta = pair[f"{col}_b"] - pair[f"{col}_a"]
            # Collapse correlated origins, rounds and seeds before resampling tasks.
            by_job = pd.DataFrame(dict(job=pair.job_id, delta=delta)).groupby("job").delta.mean()
            metrics[col] = dict(candidate_minus_baseline_ms=bootstrap_mean(by_job.to_numpy()),
                median_paired_ratio=float(np.median(pair[f"{col}_b"] / pair[f"{col}_a"])))
        pairs.append(dict(baseline=a, candidate=b, metrics=metrics))
    return dict(by_model=models, by_group=groups, paired_comparisons=pairs,
        note="Batch1, same process/hardware/precision, task-balanced shared origins, randomized model order. CI resamples whole tasks; it does not describe hardware variation.")


def combined(quality_dir, benchmark_dir, output):
    q = read_json(quality_dir / "quality_summary.json")
    e = read_json(quality_dir / "evaluation.json")
    b = read_json(benchmark_dir / "benchmark.json")
    t = read_json(benchmark_dir / "timing_summary.json")
    if e["status"] != "complete" or b["status"] != "complete":
        raise ValueError("Quality/timing runs must both be complete")
    if e["data_sha"] != b["data_sha"] or e["precision"] != b["precision"] or e["freeze_sha"] != b["freeze_sha"]:
        raise ValueError("Quality and latency data/precision/freeze differ")
    ids = lambda rows: {(r["group"], r["seed"], r["checkpoint_sha"]) for r in rows}
    if ids(e["models"]) != ids(b["models"]):
        raise ValueError("Quality and latency checkpoints differ")
    rows = []
    for group, values in q["groups"].items():
        timing = t["by_group"][group]
        rows.append(dict(group=group, full_J=values["full"].get("objective", {}).get("mean"),
            common60_J=values["common_first_60s"].get("objective", {}).get("mean"),
            pipeline_p50_ms=timing["pipeline_ms"]["p50_ms"], pipeline_p95_ms=timing["pipeline_ms"]["p95_ms"],
            forward_p50_ms=timing["forward_ms"]["p50_ms"]))
    # Show the tradeoff; never relabel 04 as an untouched holdout after this business comparison.
    for row in rows:
        j = row["common60_J"]
        row["pareto_common60_quality_pipeline_p95"] = None if j is None else not any(
            other["common60_J"] is not None and other["common60_J"] <= j
            and other["pipeline_p95_ms"] <= row["pipeline_p95_ms"]
            and (other["common60_J"] < j or other["pipeline_p95_ms"] < row["pipeline_p95_ms"])
            for other in rows)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, dict(rows=rows, hardware=b["environment"], precision=b["precision"],
        quality=q, timing=t, status="04 exploratory business comparison; no universal latency SLA was supplied"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality-dir", type=Path, required=True)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    combined(a.quality_dir, a.benchmark_dir, a.output)

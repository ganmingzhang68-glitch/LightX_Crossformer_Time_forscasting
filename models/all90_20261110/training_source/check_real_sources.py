"""Small read-only 01/02/03 source check. No training and no device04 values."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

from common import contract, digest, code_identity, write_json
from prepare import verify_schema, fuse, common_windows
from data import fit_preprocessor, Encoder


def check(root, c):
    added_root = root / "data"
    old_root = added_root / "processed_crossformer_multidevice_july_5m_10s_128features"
    verify_schema(added_root, c, c["development_devices"])
    results = []
    for dev in c["development_devices"]:
        jobs = pd.read_parquet(added_root / f"influx_jobs_added_{dev}_2026-07.parquet")
        jobs = jobs[~jobs.job_id.isin(c["excluded_jobs"])].sort_values("job_start_utc")
        jobs = jobs[(jobs.csv_end_utc - jobs.csv_start_utc).dt.total_seconds() >= 720]
        job = jobs.iloc[0]
        lo = max(job.csv_start_utc.ceil("2s"), job.job_start_utc.ceil("2s"))
        hi = lo + pd.Timedelta(seconds=718)
        raw = pd.read_parquet(added_root / f"influx_sensors_added_{dev}_2026-07.parquet",
            columns=["time_utc", "device_id", "job_id", "source_row", *c["added"]],
            filters=[("job_id", "==", job.job_id), ("time_utc", ">=", (lo-pd.Timedelta(seconds=2)).to_pydatetime()),
                     ("time_utc", "<=", hi.to_pydatetime())])
        old = pd.concat([pd.read_parquet(old_root / f"{s}.parquet",
            columns=["time_utc", "device_id", "segment_id", "observed_mask", *c["targets"]],
            filters=[("device_id", "==", dev), ("time_utc", ">=", lo.to_pydatetime()),
                     ("time_utc", "<=", hi.to_pydatetime())]) for s in ("train", "validation", "test")], ignore_index=True)
        rows = fuse(old, raw, job.job_id, c)
        origins = common_windows(rows, c)
        if not len(origins):
            raise ValueError(f"No sample windows: {dev}")
        matrix = rows[c["source_features"]].to_numpy(dtype="float32")
        repo = SimpleNamespace(c=c, records={job.job_id: {"device": dev}}, values={job.job_id: matrix})
        pre = fit_preprocessor(repo, {dev: [job.job_id]})  # encoding statistics only, not model fitting
        encoder = Encoder(c, pre)
        for name, group in c["groups"].items():
            origin = int(origins[0])
            start = origin + 1 - group["context"]
            meta = rows.iloc[start:origin+1]
            quality = np.column_stack([meta.observed_mask, meta.new_observed, meta.new_age_seconds/300.0])
            tensors = encoder.encode(matrix[start:origin+1], quality)
            assert all(np.isfinite(t).all() for t in tensors)
            future = origin + group["gap"] + 1
            assert rows.iloc[future:future+group["horizon"]].observed_mask.all()
            assert (rows.time_utc.iloc[future]-rows.time_utc.iloc[origin]).total_seconds() == 12
        results.append(dict(device=dev, job_id=job.job_id, source_rows=len(raw), fused_rows=len(rows),
            common_windows=len(origins), raw_fields=matrix.shape[1], added_fields=len(c["added"]),
            added_missing={f: int(rows[f].isna().sum()) for f in c["added"] if rows[f].isna().any()},
            groups_checked=list(c["groups"]), all_groups_encoded_finite=True))
        print(dev, "read-only source check passed", flush=True)
    return dict(status="passed", scope="one 12-minute sample per development device; not full-dataset validation",
        contract_sha=digest(c), code_sha=code_identity(), samples=results, training_started=False,
        device04_values_read=False, source_files_modified=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contract", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output, check(a.root.resolve(), contract(a.contract)))

"""Read-only source fusion; writes a new job-sharded dataset, never trains."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import contract, digest, file_hash, read_json, safe_new_output, write_json


def numeric_snapshot(raw, c):
    """All 90 fields retained; date/result validity checked at each source row."""
    raw = raw.copy()
    source = pd.to_datetime(raw.time_utc, utc=True)
    date_valid = {}
    for field in c["dates"]:
        # The two date strings use the CSV's documented Beijing wall clock.
        date = pd.to_datetime(raw[field], errors="coerce").dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
        valid = date.notna() & (date <= source)
        date_valid[field] = valid
        age_days = (source - date).dt.total_seconds() / 86400.0
        raw[field] = np.log1p(age_days.where(valid))
    for result, date in c["result_dates"].items():
        raw[result] = pd.to_numeric(raw[result], errors="raise").where(date_valid[date])
    for field in c["added"]:
        raw[field] = pd.to_numeric(raw[field], errors="raise")
        if np.isinf(raw[field].to_numpy(dtype="float64")).any():
            raise ValueError(f"Infinite source values: {field}")
    return raw


def regularize(raw, c, delay_seconds=0):
    raw = numeric_snapshot(raw, c)
    raw["source_time_utc"] = pd.to_datetime(raw.time_utc, utc=True)
    raw["available_time_utc"] = raw.source_time_utc + pd.Timedelta(seconds=delay_seconds)
    raw = raw.sort_values(["source_time_utc", "source_row"], kind="stable")
    raw = raw.drop_duplicates("source_time_utc", keep="last")
    raw["time_utc"] = raw.available_time_utc.dt.ceil("2s")
    raw = raw.drop_duplicates("time_utc", keep="last")
    chunks = (raw.time_utc.diff().dt.total_seconds().gt(4)
              | raw.source_time_utc.diff().dt.total_seconds().gt(4)).cumsum()
    pieces = []
    for chunk_id, part in raw.groupby(chunks, sort=True):
        part = part.set_index("time_utc")
        grid = pd.date_range(part.index[0], part.index[-1], freq="2s")
        z = part[["source_time_utc", "available_time_utc", "source_row", *c["added"]]].reindex(grid)
        observed = z.source_time_utc.notna()
        # Copy a whole previous snapshot ONLY into one inserted grid row.
        # Never forward-fill an explicit null in a real snapshot (e.g. invalid test result).
        held = z.ffill(limit=1)
        z.loc[~observed] = held.loc[~observed]
        z["new_observed"] = observed.to_numpy()
        z["new_chunk"] = int(chunk_id)
        z["new_age_seconds"] = (z.index - z.source_time_utc).dt.total_seconds()
        z.index.name = "time_utc"
        pieces.append(z.reset_index())
    if not pieces:
        raise ValueError("Empty added job")
    return pd.concat(pieces, ignore_index=True)


def fuse(old, raw, job_id, c, delay_seconds=0):
    new = regularize(raw, c, delay_seconds)
    old = old.sort_values("time_utc", kind="stable").copy()
    if old.duplicated("time_utc").any():
        raise ValueError("Duplicate old timestamps / overlapping old splits")
    old.loc[~old.observed_mask.astype(bool), c["targets"]] = np.nan
    old[c["targets"]] = old.groupby("segment_id", sort=False)[c["targets"]].ffill(limit=1)
    old = old.rename(columns={"segment_id": "old_segment"})
    z = old.merge(new, on="time_utc", how="inner", validate="one_to_one")
    z = z.sort_values("time_utc").reset_index(drop=True)
    # Missing added fields remain representable through masks, including platform-on-position.
    good = np.isfinite(z[c["targets"]].to_numpy(dtype="float32")).all(axis=1)
    z = z.loc[good].copy()
    boundary = (z.time_utc.diff().dt.total_seconds().ne(2)
                | z.old_segment.ne(z.old_segment.shift()) | z.new_chunk.ne(z.new_chunk.shift()))
    z["segment_id"] = boundary.cumsum().astype("int32")
    z["job_id"] = job_id
    if (z.available_time_utc > z.time_utc).any():
        raise ValueError("Future source leaked into history")
    return z.reset_index(drop=True)


def common_windows(rows, c, stride=5):
    """Identical forecast origins, using longest history and horizon of all groups."""
    if stride < 1:
        raise ValueError("Positive stride required")
    history = max(g["context"] for g in c["groups"].values())
    support = max(g["gap"] + g["horizon"] for g in c["groups"].values())
    observed = rows.observed_mask.to_numpy(bool)
    bad_prefix = np.r_[0, np.cumsum(~observed)]
    origins = []
    for _, part in rows.groupby("segment_id", sort=False):
        lo, hi = int(part.index[0]), int(part.index[-1]) + 1
        candidates = np.arange(lo + history - 1, hi - support, stride, dtype=np.int64)
        valid = np.ones(len(candidates), dtype=bool)
        for g in c["groups"].values():
            future = candidates + g["gap"] + 1
            end = future + g["horizon"]
            valid &= bad_prefix[end] == bad_prefix[future]
        origins.extend(candidates[valid].tolist())
    return np.asarray(origins, dtype=np.int64)


def verify_schema(added_root, c, devices):
    meta = {"time_utc", "time_beijing", "device_id", "job_id", "source_file", "source_row"}
    for dev in devices:
        names = set(pq.ParquetFile(added_root / f"influx_sensors_added_{dev}_2026-07.parquet").schema_arrow.names) - meta
        if names - set(c["targets"]) != set(c["added"]):
            raise ValueError(f"New90 source schema differs from docs: {dev}")


def prepare(args):
    c = contract(args.contract)
    devices = [c["external_device"]] if args.external else c["development_devices"]
    verify_schema(args.added_root, c, devices)
    old_manifest = read_json(args.old_root / "feature_manifest.json")
    if old_manifest["feature_order"] != c["targets"] or old_manifest.get("continuous_values_clipped", False):
        raise ValueError("Old source order/clipping contract differs")
    out = safe_new_output(args.output, [args.old_root, args.added_root])
    out.mkdir(parents=True)
    manifest = dict(status="building", contract_sha=digest(c), external_only=args.external,
        devices=devices, stride=args.stride, added_delay_seconds=args.added_delay_seconds,
        source_features=c["source_features"], groups=c["groups"], jobs=[], sources={},
        cohort="intersection of eligible forecast origins for all configured groups",
        availability="source snapshot plus simulated delay; actual arrival time unavailable",
        date_encoding="log1p days since test as of source snapshot; future/invalid dates and paired results masked",
        fill="whole added snapshot held one missing tick; old legacy interpolations replaced by one causal hold")
    old_paths = [args.old_root / f"{split}.parquet" for split in ("train", "validation", "test")]
    manifest["old_sources"] = {str(p): file_hash(p) for p in [args.old_root / "feature_manifest.json", *old_paths]}
    write_json(out / "manifest.json", manifest)
    for dev in devices:
        source_path = args.added_root / f"influx_sensors_added_{dev}_2026-07.parquet"
        jobs_path = args.added_root / f"influx_jobs_added_{dev}_2026-07.parquet"
        jobs = pd.read_parquet(jobs_path).sort_values("job_start_utc")
        if jobs.job_id.duplicated().any() or set(jobs.device_id) != {dev}:
            raise ValueError("Jobs identity is invalid")
        audit = []
        manifest["sources"][dev] = {str(p): file_hash(p) for p in (source_path, jobs_path)}
        for j in jobs.itertuples(index=False):
            if j.job_id in c["excluded_jobs"]:
                audit.append(dict(job_id=j.job_id, reason="pre-existing old-source gap exclusion"))
                continue
            raw = pd.read_parquet(source_path, columns=["time_utc", "device_id", "job_id", "source_row", *c["added"]],
                                  filters=[("job_id", "==", j.job_id)])
            if raw.empty or set(raw.device_id) != {dev} or set(raw.job_id) != {j.job_id}:
                raise ValueError("Source task identity mismatch")
            lo = max(raw.time_utc.min().floor("2s"), j.job_start_utc.ceil("2s"))
            hi = min(raw.time_utc.max().ceil("2s") + pd.Timedelta(seconds=args.added_delay_seconds),
                     j.job_end_utc.floor("2s"))
            overlap = jobs[(jobs.job_start_utc <= hi) & (jobs.job_end_utc >= lo)]
            if set(overlap.job_id) - {j.job_id}:
                raise ValueError("Ambiguous overlapping task join")
            old_parts = [pd.read_parquet(p, columns=["time_utc", "device_id", "segment_id", "observed_mask", *c["targets"]],
                filters=[("device_id", "==", dev), ("time_utc", ">=", lo.to_pydatetime()),
                         ("time_utc", "<=", hi.to_pydatetime())]) for p in old_paths]
            if not any(len(part) for part in old_parts):
                audit.append(dict(job_id=j.job_id, reason="no old rows"))
                continue
            rows = fuse(pd.concat(old_parts, ignore_index=True), raw, j.job_id, c, args.added_delay_seconds)
            origins = common_windows(rows, c, args.stride)
            audit.append(dict(job_id=j.job_id, raw_rows=len(raw), fused_rows=len(rows), windows=len(origins),
                added_missing={f: int(rows[f].isna().sum()) for f in c["added"]}))
            if not len(origins):
                continue
            rel = Path(dev) / j.job_id
            dest = out / rel
            dest.mkdir(parents=True)
            np.save(dest / "values.npy", rows[c["source_features"]].to_numpy(dtype="float32"))
            np.save(dest / "origins.npy", origins)
            rows[["time_utc", "segment_id", "observed_mask", "new_observed", "new_age_seconds",
                  "source_time_utc", "available_time_utc", "source_row"]].to_parquet(dest / "rows.parquet", index=False)
            record = dict(device=dev, job_id=j.job_id, path=rel.as_posix(), rows=len(rows), windows=len(origins),
                          start=rows.time_utc.iloc[0].isoformat(), end=rows.time_utc.iloc[-1].isoformat(),
                          hashes={n: file_hash(dest / n) for n in ("values.npy", "origins.npy", "rows.parquet")})
            manifest["jobs"].append(record)
            write_json(out / "manifest.json", manifest)
            print(dev, j.job_id, "rows", len(rows), "common windows", len(origins), flush=True)
        write_json(out / dev / "alignment.json", audit)
        if not any(j["device"] == dev for j in manifest["jobs"]):
            raise ValueError(f"No usable jobs for {dev}")
    manifest["status"] = "complete"
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-root", type=Path, required=True)
    p.add_argument("--added-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contract", type=Path)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--added-delay-seconds", type=int, default=0)
    p.add_argument("--external", action="store_true")
    a = p.parse_args()
    if a.stride < 1 or a.added_delay_seconds < 0:
        p.error("stride >= 1 and delay >= 0 required")
    prepare(a)

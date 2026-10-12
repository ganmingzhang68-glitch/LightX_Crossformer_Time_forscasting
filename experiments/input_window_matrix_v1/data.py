"""Shared origins, whole-job splits, train-only encoding, all 90 known masks."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import digest, file_hash, read_json, validate_selection
from prepare import common_windows


class Repository:
    def __init__(self, root, c, external=False, verify_hashes=True):
        self.root, self.c = Path(root), c
        self.manifest = read_json(self.root / "manifest.json")
        m = self.manifest
        expected = [c["external_device"]] if external else c["development_devices"]
        if (m["status"] != "complete" or m["contract_sha"] != digest(c)
                or m["devices"] != expected or m["external_only"] != external):
            raise ValueError("Incomplete/wrong contract or device isolation")
        if m["source_features"] != c["source_features"] or m["groups"] != c["groups"]:
            raise ValueError("Declared source order or window groups differ from storage contract")
        self.records, self.values, self.rows, self.origins = {}, {}, {}, {}
        for record in m["jobs"]:
            job = record["job_id"]
            if job in self.records or record["device"] not in expected:
                raise ValueError("Duplicate job or unexpected device")
            dest = (self.root / record["path"]).resolve()
            if self.root.resolve() not in dest.parents:
                raise ValueError("Job path escapes dataset")
            if verify_hashes:
                for name, expected_hash in record["hashes"].items():
                    if file_hash(dest / name) != expected_hash:
                        raise ValueError(f"Changed data file: {dest / name}")
            values = np.load(dest / "values.npy", mmap_mode="r", allow_pickle=False)
            rows = pd.read_parquet(dest / "rows.parquet")
            origins = np.load(dest / "origins.npy", allow_pickle=False)
            if values.shape != (len(rows), 218) or len(rows) != record["rows"]:
                raise ValueError("Wrong 218-field matrix dimensions")
            if rows.time_utc.duplicated().any() or not rows.time_utc.is_monotonic_increasing:
                raise ValueError("Invalid task timestamps")
            if rows.groupby("segment_id").time_utc.diff().dt.total_seconds().dropna().ne(2).any():
                raise ValueError("Non-2s segment")
            if (rows.available_time_utc > rows.time_utc).any():
                raise ValueError("Future source")
            if not np.array_equal(origins, common_windows(rows, c, m["stride"])) or len(origins) != record["windows"]:
                raise ValueError("Window boundaries or observed targets invalid")
            for start in range(0, len(values), 32768):
                block = values[start:start + 32768]
                if not np.isfinite(block[:, :128]).all() or np.isinf(block).any():
                    raise ValueError("Invalid numeric matrix")
            self.records[job], self.values[job], self.rows[job], self.origins[job] = record, values, rows, origins
        if {r["device"] for r in self.records.values()} != set(expected):
            raise ValueError("A required device has no eligible jobs")

    def jobs(self, dev):
        return [r["job_id"] for r in sorted(self.records.values(), key=lambda r: r["start"])
                if r["device"] == dev and len(self.origins[r["job_id"]])]


def split_jobs(repo):
    train, validation, refit = {}, {}, {}
    for dev in repo.c["development_devices"]:
        jobs = repo.jobs(dev)
        if len(jobs) < 3:
            raise ValueError(f"At least 3 eligible jobs required: {dev}")
        count = max(1, int(np.ceil(len(jobs) * repo.c["selection"]["validation_job_fraction"])))
        train[dev], validation[dev], refit[dev] = jobs[:-count], jobs[-count:], jobs
    return dict(train=train, validation=validation, refit=refit)


def fit_preprocessor(repo, selection):
    c = repo.c
    validate_selection(selection, c)
    n = 218
    count, mean, m2 = np.zeros(n), np.zeros(n), np.zeros(n)
    minimum, maximum = np.full(n, np.inf), np.full(n, -np.inf)
    vocab = {f: set() for f in c["categorical"]}
    for dev, jobs in selection.items():
        for job in jobs:
            if repo.records[job]["device"] != dev:
                raise ValueError("Training job/device identity mismatch")
            values = repo.values[job]
            for start in range(0, len(values), 32768):
                x = np.asarray(values[start:start + 32768], dtype="float64")
                valid = np.isfinite(x)
                nc = valid.sum(0)
                total = count + nc
                block_mean = np.divide(np.where(valid, x, 0).sum(0), nc, out=np.zeros(n), where=nc > 0)
                delta = block_mean - mean
                block_m2 = (np.where(valid, x - block_mean, 0) ** 2).sum(0)
                m2 += block_m2 + np.divide(delta ** 2 * count * nc, total, out=np.zeros(n), where=total > 0)
                mean += np.divide(delta * nc, total, out=np.zeros(n), where=total > 0)
                count = total
                minimum = np.minimum(minimum, np.where(valid, x, np.inf).min(0))
                maximum = np.maximum(maximum, np.where(valid, x, -np.inf).max(0))
                for field in vocab:
                    i = c["source_features"].index(field)
                    vocab[field].update(x[valid[:, i], i].tolist())
    if (count[:128] == 0).any():
        raise ValueError("Missing training target")
    scale = np.sqrt(np.divide(m2, count, out=np.zeros(n), where=count > 0))
    constant = scale < 1e-6
    scale[constant] = 1.0
    # Retain absolute levels for constant new settings; zero variance is not a field-selection rule.
    const_added = constant.copy()
    const_added[:128] = False
    scale[const_added] = np.maximum(np.abs(mean[const_added]), 1.0)
    mean[const_added] = 0.0
    active = [f for i, f in enumerate(c["targets"]) if maximum[i] > minimum[i]]
    return dict(mean=mean.tolist(), scale=scale.tolist(), count=count.astype(int).tolist(),
        categorical_vocabulary={f: sorted(v) for f, v in vocab.items()}, active_targets=active,
        active_source_fields=[f for i, f in enumerate(c["source_features"]) if maximum[i] > minimum[i]],
        constant_fields=[f for i, f in enumerate(c["source_features"]) if constant[i]],
        never_observed_fields=[f for i, f in enumerate(c["source_features"]) if count[i] == 0],
        fit_selection=selection, fit_selection_sha=digest(selection), contract_sha=digest(c))


class Encoder:
    def __init__(self, c, pre, profile):
        if pre["contract_sha"] != digest(c):
            raise ValueError("Encoder contract mismatch")
        self.c, self.pre, self.profile = c, pre, profile
        self.mean, self.scale = np.asarray(pre["mean"], dtype="float32"), np.asarray(pre["scale"], dtype="float32")
        if self.mean.shape != (218,) or self.scale.shape != (218,) or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid scaler")
        self.dynamic_indices = [c["source_features"].index(f) for f in profile["dynamic"]]
        self.condition_indices = [c["source_features"].index(f) for f in profile["conditions"]]
        self.mask_indices = [c["source_features"].index(f) for f in c["added"] if f in set(profile["dynamic"] + profile["conditions"])]
        self.event_fields = c["boolean"] + profile["conditions"] if profile["events"] else []
        self.aux_width = (len(self.mask_indices) if profile["missing_masks"] else 0) + (3 if profile["quality"] else 0) + 4 * len(self.event_fields)

    def encode(self, raw, quality):
        raw = np.asarray(raw, dtype="float32")
        known = np.isfinite(raw)
        x = np.where(known, (raw - self.mean) / self.scale, 0.0).astype("float32")
        conditions = x[:, self.condition_indices].copy()
        for f in set(self.c["categorical"]) & set(self.profile["conditions"]):
            j = self.profile["conditions"].index(f)
            source = self.c["source_features"].index(f)
            v = raw[:, source]
            tokens = np.where(known[:, source], 1, 0).astype("float32")  # missing=0; unknown=1
            for token, value in enumerate(self.pre["categorical_vocabulary"][f], 2):
                tokens[v == value] = token
            conditions[:, j] = tokens
        parts = []
        if self.profile["missing_masks"]:
            parts.append(known[:, self.mask_indices].astype("float32"))
        if self.profile["quality"]:
            parts.append(np.asarray(quality, dtype="float32"))
        if self.event_fields:
            indices = [self.c["source_features"].index(f) for f in self.event_fields]
            observed = np.column_stack([quality[:, 0] if f in self.c["targets"] else quality[:, 1] for f in self.event_fields])
            parts.append(event_channels(raw[:, indices], observed=observed.astype(bool) & known[:, indices]))
        aux = np.concatenate(parts, axis=1) if parts else np.empty((len(raw), 0), dtype="float32")
        dynamic = x[:, self.dynamic_indices]
        if not all(np.isfinite(v).all() for v in (dynamic, conditions, aux)):
            raise ValueError("Invalid encoded input")
        return dynamic, conditions, aux


class Windows(Dataset):
    def __init__(self, repo, selection, group, pre, profile, balance=False, cap=0):
        self.repo, self.c, self.group = repo, repo.c, group
        self.g = self.c["groups"][group]
        self.profile = profile
        self.encoder = Encoder(self.c, pre, profile)
        self.target_indices = [self.c["source_features"].index(f) for f in profile["targets"]]
        by_device = {}
        for dev, jobs in selection.items():
            values = []
            for job in jobs:
                if repo.records[job]["device"] != dev:
                    raise ValueError("Window selection device mismatch")
                values.extend((dev, job, int(origin)) for origin in repo.origins[job])
            if not values:
                raise ValueError("Empty window selection")
            by_device[dev] = values
        limit = min(len(v) for v in by_device.values()) if balance else 0
        if cap > 0:
            limit = min(limit, cap) if limit else cap
        self.items, self.counts = [], {}
        for dev, values in by_device.items():
            indices = np.linspace(0, len(values) - 1, limit, dtype=np.int64) if 0 < limit < len(values) else range(len(values))
            chosen = [values[i] for i in indices]
            self.items.extend(chosen)
            self.counts[dev] = len(chosen)

    def __len__(self):
        return len(self.items)

    def raw_input(self, index):
        _, job, origin = self.items[index]
        start = origin + 1 - self.g["context"]
        meta = self.repo.rows[job].iloc[start:origin + 1]
        quality = np.column_stack([meta.observed_mask, meta.new_observed, meta.new_age_seconds / 300.0]).astype("float32")
        raw = np.asarray(self.repo.values[job][start:origin + 1], dtype="float32")
        return raw, quality

    def __getitem__(self, index):
        dev, job, origin = self.items[index]
        raw, quality = self.raw_input(index)
        dynamic, conditions, aux = self.encoder.encode(raw, quality)
        future = origin + self.g["gap"] + 1
        target = np.asarray(self.repo.values[job][future:future + self.g["horizon"]], dtype="float32")[:, self.target_indices]
        target = (target - self.encoder.mean[self.target_indices]) / self.encoder.scale[self.target_indices]
        if not np.isfinite(target).all():
            raise ValueError("Missing target")
        return tuple(torch.from_numpy(v.copy()) for v in (dynamic, conditions, aux, target)) + (dev + "|" + job,)


def event_channels(raw, observed):
    """Historical edges only; first sample/held snapshots have unknown edge age."""
    known = np.isfinite(raw) & observed
    valid = known.copy()
    valid[0] = False
    valid[1:] &= known[:-1]
    delta = np.zeros_like(raw)
    np.subtract(raw[1:], raw[:-1], out=delta[1:], where=valid[1:])
    changed = (delta != 0) & valid
    indices = np.arange(len(raw))[:, None]
    last = np.maximum.accumulate(np.where(changed, indices, -1), axis=0)
    unknown = np.maximum.accumulate(np.where(~valid, indices, -1), axis=0)
    age_known = (last >= 0) & (last > unknown)
    age = np.where(age_known, (indices - last) * 2 / 300.0, 0)
    return np.concatenate([(delta > 0).astype("float32"), (delta < 0).astype("float32"), age, age_known], axis=1).astype("float32")


def apply_shared_cohort(repo, profiles):
    """One intersection across every input, target set and all four windows."""
    c = repo.c
    required_inputs = sorted({f for p in profiles.values() if not p["missing_masks"]
                              for f in p["dynamic"] + p["conditions"] if f not in c["targets"]})
    added_targets = sorted({f for p in profiles.values() for f in p["targets"] if f not in c["targets"]})
    ii = [c["source_features"].index(f) for f in required_inputs]
    ti = [c["source_features"].index(f) for f in added_targets]
    longest = max(g["context"] for g in c["groups"].values())
    counts = {}
    for job, values in repo.values.items():
        origins = repo.origins[job]
        input_good = np.isfinite(values[:, ii]).all(1) if ii else np.ones(len(values), dtype=bool)
        prefix = np.r_[0, np.cumsum(~input_good)]
        keep = prefix[origins + 1] == prefix[origins + 1 - longest]
        if ti:
            # New5 labels must be observed snapshots, not forward held values.
            target_good = np.isfinite(values[:, ti]).all(1) & repo.rows[job].new_observed.to_numpy(bool)
            bp = np.r_[0, np.cumsum(~target_good)]
            for g in c["groups"].values():
                future = origins + g["gap"] + 1
                keep &= bp[future + g["horizon"]] == bp[future]
        repo.origins[job] = origins[keep]
        counts[job] = len(repo.origins[job])
    cohort = dict(required_history_fields=required_inputs, observed_added_targets=added_targets,
        windows_by_job=counts, origins_sha=digest({job: x.tolist() for job, x in repo.origins.items()}))
    if not sum(counts.values()):
        raise ValueError("No shared origins satisfy selected profiles")
    return cohort

"""Frozen contracts and identities for the all-90 latency experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def code_identity():
    return digest({p.relative_to(HERE).as_posix(): file_hash(p) for p in sorted(HERE.rglob("*.py"))})


def contract(path=None):
    c = read_json(path or HERE / "contract.json")
    if len(c["targets"]) != 128 or len(c["added"]) != 90:
        raise ValueError("Exactly 128 targets and ALL 90 added fields are required")
    if c["source_features"] != c["targets"] + c["added"] or len(set(c["source_features"])) != 218:
        raise ValueError("The ordered 218-field contract is invalid")
    if set(c["continuous"]) & set(c["boolean"]) or set(c["continuous"] + c["boolean"]) != set(c["targets"]):
        raise ValueError("Target type contract is invalid")
    if len(c["continuous"]) != 56 or len(c["boolean"]) != 72:
        raise ValueError("Expected original 56 continuous and 72 boolean targets")
    if set(c["categorical"] + c["dates"]) - set(c["added"]):
        raise ValueError("Unknown encoded fields")
    if c["external_device"] in c["development_devices"]:
        raise ValueError("Device04 isolation violation")
    for name, group in c["groups"].items():
        for key in ("context", "horizon", "gap"):
            if type(group[key]) is not int or group[key] < (0 if key == "gap" else 1):
                raise ValueError(f"Invalid window: {name}/{key}")
        if group["gap"] != 5 or c["cadence_seconds"] != 2:
            raise ValueError("This experiment requires a 10-second gap and 2-second cadence")
    return c


def safe_new_output(path, protected=()):
    p = Path(path).resolve()
    for source in [HERE, *(Path(x).resolve() for x in protected)]:
        if p == source or p in source.parents or source in p.parents:
            raise ValueError(f"Output overlaps source/code: {p}")
    if p.exists():
        raise FileExistsError(p)
    return p


def validate_selection(selection, c):
    if not selection or not set(selection) <= set(c["development_devices"]):
        raise ValueError("Fitting may use only development devices 01/02/03")
    if any(not jobs for jobs in selection.values()):
        raise ValueError("Empty training job selection")

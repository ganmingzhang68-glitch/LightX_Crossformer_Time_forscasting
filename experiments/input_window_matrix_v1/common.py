"""Frozen218 storage contract and identities for the input/window matrix."""
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
    paths = list(HERE.rglob("*.py")) + [HERE / "contract.json", HERE / "catalog.json"]
    return digest({p.relative_to(HERE).as_posix(): file_hash(p) for p in sorted(paths)})


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


def catalog():
    c, result = contract(), read_json(HERE / "catalog.json")
    if result["provenance"]["storage_contract_sha"] != digest(c):
        raise ValueError("Historical catalog/storage contract mismatch")
    for name, p in result["profiles"].items():
        if len(set(p["dynamic"])) != len(p["dynamic"]) or len(set(p["conditions"])) != len(p["conditions"]):
            raise ValueError("Duplicate input fields: " + name)
        if set(p["dynamic"] + p["conditions"]) - set(c["source_features"]):
            raise ValueError("Unknown input field: " + name)
        if set(p["targets"]) - set(p["dynamic"]):
            raise ValueError("Targets must have dynamic history for persistence/residual: " + name)
        if len(p["targets"]) != p["output_count"] or len(set(p["dynamic"] + p["conditions"])) != p["raw_input_count"]:
            raise ValueError("Invalid profile dimensions: " + name)
        if set(p["continuous"]) & set(p["boolean"]) or set(p["continuous"] + p["boolean"]) != set(p["targets"]):
            raise ValueError("Invalid target types: " + name)
    return result


def selected_profiles(names=None, output_policy="old128"):
    import copy
    c, cat = contract(), catalog()
    if output_policy not in ("historical", "old128"):
        raise ValueError("Unknown output policy")
    names = list(names) if names else [n for n, p in cat["profiles"].items() if p["default"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Require unique profile names")
    chosen, identities, aliases = {}, {}, {}
    for name in names:
        if name not in cat["profiles"]:
            raise ValueError("Unknown profile: " + name)
        p = copy.deepcopy(cat["profiles"][name])
        if output_policy == "old128":
            p.update(targets=c["targets"], continuous=c["continuous"], boolean=c["boolean"], output_count=128)
        signature = digest({k: p[k] for k in ("dynamic", "conditions", "targets", "quality", "missing_masks", "events")})
        if signature in identities:
            aliases[name] = identities[signature]
        else:
            identities[signature], chosen[name] = name, p
    return chosen, aliases


def runtime_contract(c, profile):
    import copy
    result = copy.deepcopy(c)
    result.update(targets=profile["targets"], continuous=profile["continuous"], boolean=profile["boolean"], profile=profile,
        score_targets=[f for f in profile["targets"] if f in c["targets"]],
        storage_boolean=c["boolean"],
        storage_contract_sha=digest(c))
    return result


def verify_signed(value, key):
    if digest({k: v for k, v in value.items() if k != key}) != value[key]:
        raise ValueError("Signature mismatch: " + key)

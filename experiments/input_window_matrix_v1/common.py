"""Frozen218 storage contract and identities for the input/window matrix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

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


def validate_profile(name, p, c):
    required = {"dynamic", "conditions", "targets", "continuous", "boolean", "quality",
                "missing_masks", "events", "raw_input_count", "output_count"}
    if required - set(p):
        raise ValueError("Incomplete profile: " + name)
    if not isinstance(p["dynamic"], list) or not isinstance(p["conditions"], list):
        raise ValueError("Profile fields must be ordered lists: " + name)
    if any(not isinstance(f, str) or not f for f in p["dynamic"] + p["conditions"]):
        raise ValueError("Invalid input field name: " + name)
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
    if any(type(p[k]) is not bool for k in ("quality", "missing_masks", "events")):
        raise ValueError("Profile encoder switches must be booleans: " + name)


def catalog():
    c, result = contract(), read_json(HERE / "catalog.json")
    if result["provenance"]["storage_contract_sha"] != digest(c):
        raise ValueError("Historical catalog/storage contract mismatch")
    for name, p in result["profiles"].items():
        validate_profile(name, p, c)
    return result


def load_feature_manifests(paths):
    """Load strict, ordered custom profiles without changing the frozen218 storage contract."""
    c, profiles, sources = contract(), {}, []
    allowed = {"version", "name", "features", "dynamic", "conditions", "quality",
               "missing_masks", "events", "note"}
    for source in paths or []:
        path = Path(source).resolve()
        document = read_json(path)
        if not isinstance(document, dict) or set(document) - allowed:
            raise ValueError("Unknown custom feature manifest keys: " + str(path))
        if document.get("version") != 1:
            raise ValueError("Custom feature manifest version must be 1: " + str(path))
        name = document.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("Custom profile name must be path-safe ASCII: " + str(path))
        if name in profiles:
            raise ValueError("Duplicate custom profile name: " + name)
        simple = "features" in document
        advanced = "dynamic" in document or "conditions" in document
        if simple == advanced:
            raise ValueError("Use exactly one of features or dynamic/conditions: " + name)
        if simple:
            dynamic, conditions = document["features"], []
        else:
            dynamic, conditions = document.get("dynamic"), document.get("conditions", [])
        switches = {k: document.get(k, False) for k in ("quality", "missing_masks", "events")}
        profile = dict(dynamic=dynamic, conditions=conditions, targets=list(c["targets"]),
            continuous=list(c["continuous"]), boolean=list(c["boolean"]), **switches,
            raw_input_count=len(set(dynamic + conditions)) if isinstance(dynamic, list) and isinstance(conditions, list) else -1,
            output_count=128, evidence="user_feature_manifest", default=False,
            note=document.get("note", "Custom ordered input contract."))
        validate_profile(name, profile, c)
        profiles[name] = profile
        sources.append({"path": str(path), "sha256": file_hash(path)})
    return profiles, sources


def selected_profiles(names=None, output_policy="old128", custom_profiles=None):
    import copy
    c, cat = contract(), catalog()
    if output_policy not in ("historical", "old128"):
        raise ValueError("Unknown output policy")
    custom_profiles = copy.deepcopy(custom_profiles or {})
    if set(custom_profiles) & set(cat["profiles"]):
        raise ValueError("Custom profile name conflicts with catalog")
    for name, p in custom_profiles.items():
        validate_profile(name, p, c)
    names = list(names) if names else ([] if custom_profiles else [n for n, p in cat["profiles"].items() if p["default"]])
    if (not names and not custom_profiles) or len(set(names)) != len(names):
        raise ValueError("Require unique profile names")
    chosen, identities, aliases = {}, {}, {}
    candidates = []
    for name in names:
        if name not in cat["profiles"]:
            raise ValueError("Unknown profile: " + name)
        candidates.append((name, copy.deepcopy(cat["profiles"][name])))
    candidates.extend((name, p) for name, p in custom_profiles.items())
    for name, p in candidates:
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

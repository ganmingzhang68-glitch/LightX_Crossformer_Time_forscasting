"""Explicit one-time snapshot of historical PLC input/output contracts. Never trains."""
from pathlib import Path
import argparse
from common import contract, digest, file_hash, read_json, write_json

HERE = Path(__file__).resolve().parent


def build(workspace):
    c = contract()
    source = workspace / "experiments/new90_to128_ampfix_v1/configs/catalog.json"
    old = read_json(source)
    selected_path = HERE.parent / "latency_abcd_27/C_129_history_plus_4_conditions_to_27/selected_targets_27.json"
    selected = read_json(selected_path)
    target27 = selected["continuous_features"] + selected["boolean_features"]
    base5 = old["base_added"]
    profiles = {}

    def add(name, dynamic, conditions, targets, quality=False, masks=False, events=False,
            evidence="completed_training", default=True, note=""):
        profiles[name] = dict(dynamic=dynamic, conditions=conditions, targets=targets,
            continuous=[f for f in targets if f in c["continuous"] or f in base5],
            boolean=[f for f in targets if f in c["boolean"]],
            quality=quality, missing_masks=masks, events=events,
            raw_input_count=len(set(dynamic + conditions)), output_count=len(targets),
            evidence=evidence, default=default, note=note)

    add("BASE128", c["targets"], [], c["targets"], note="Published old128 baseline input/output; new window retraining, not Epoch45 weights.")
    add("A128_27", c["targets"], [], target27)
    add("B133_27", c["targets"] + base5, [], target27)
    add("C129_4_27", c["targets"] + ["打印层数"], base5[1:], target27)
    add("D129_5_27", c["targets"] + ["打印层数"], base5[1:] + ["设风速"], target27,
        note="Set wind speed may proxy device identity; inspect per-device quality.")
    add("F133_133", c["targets"] + base5, [], c["targets"] + base5,
        evidence="historical_code_pending_result_confirmation", default=False,
        note="Extra5 supervised as historical numeric auxiliary outputs (including categorical mode); primary J still uses old128.")
    for name, v in old["variants"].items():
        event = v["events"]
        add("N90_" + name, v["dynamic"], v["conditions"], c["targets"], quality=True,
            events=event, default=not event,
            evidence="code_only_no_completed_result_found" if event else "completed_screen_or_confirmation",
            note="Old G1..G6 names denote feature groups, not the new time-window G1..G4.")
    add("ALL90", c["targets"] + c["dynamic_added"], c["added"], c["targets"],
        quality=True, masks=True, note="All90 uses per-field masks; includes constants and historically unavailable dates/results.")
    for name, p in profiles.items():
        if set(p["dynamic"] + p["conditions"] + p["targets"]) - set(c["source_features"]):
            raise ValueError("Unknown historical field: " + name)
    return dict(version=1, profiles=profiles, base_added=base5,
        provenance={"new90_catalog_sha256": file_hash(source), "selected27_sha256": file_hash(selected_path),
                    "storage_contract_sha": digest(c)},
        protocol="Retrain historical business field contracts in a unified pipeline; no historical weights reused.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, default=HERE / "catalog.json")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output, build(a.workspace))

"""Rebuild the contract from the reviewed HTML or its bundled field-only extract."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import json

from common import HERE, digest, file_hash, read_json, write_json


class SnapshotParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.active = attrs.get("id") == "data-app-reviewed-snapshot" or "data-app-snapshot-chunk" in attrs

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = False

    def handle_data(self, text):
        if self.active:
            self.chunks.append(text)


def build(root):
    root = Path(root)
    source = read_json(HERE / "sources/added_fields.json")
    audit = source["fields"]
    doc = root / "docs/PLC_NEW90_TO_128_EXPERIMENT_PLAN.html"
    if doc.exists():
        expected = next(value for name, value in source["provenance"].items()
                        if name.replace("\\", "/") == "docs/PLC_NEW90_TO_128_EXPERIMENT_PLAN.html")
        if file_hash(doc) != expected:
            raise ValueError("Reviewed HTML source has changed")
        parser = SnapshotParser()
        parser.feed(doc.read_text(encoding="utf-8"))
        snapshot = json.loads("".join(parser.chunks))
        extracted = [{key: row[key] for key in ("field", "tier", "caution")}
                     for row in snapshot["queries"]["fields"]["rows"]]
        if extracted != audit:
            raise ValueError("Bundled fields differ from reviewed HTML")
    old_path = root / "LightX_Crossformer_Time_forscasting/artifacts/feature_manifest.json"
    if not old_path.exists():
        old_path = root / "artifacts/feature_manifest.json"
    old = read_json(old_path)
    # Compare parsed content so Windows/Linux checkout line endings do not alter identity.
    if digest(old) != source["feature_manifest_content_sha256"]:
        raise ValueError("Original 128-field manifest has changed")
    targets = old["feature_order"]
    added = [r["field"] for r in audit]
    if len(added) != 90 or len(set(targets + added)) != 218:
        raise ValueError("Document is not the expected 90-field source")
    categories = ["设铺粉模式", "负压测试结果", "正压测试结果"]
    dates = ["负压测试日期", "正压测试日期"]
    lineage = []
    for r in audit:
        field = r["field"]
        kind = "category_train_vocabulary" if field in categories else "numeric_train_scale"
        if field in dates:
            kind = "log1p_age_days_as_of_history_tick"
        lineage.append(dict(field=field, included=True, encoding=kind,
                            missing="neutral value plus explicit per-field known mask",
                            prior_caution=r["caution"], prior_tier=r["tier"]))
    return dict(version=1, source_features=targets + added, targets=targets, added=added,
        continuous=old["continuous_features"], boolean=old["boolean_features"],
        categorical=categories, dates=dates, dynamic_added=["打印层数", "铺粉器速度"],
        result_dates={"负压测试结果": "负压测试日期", "正压测试结果": "正压测试日期"},
        cadence_seconds=2, groups={
            "G1_300_10_120": dict(context=150, gap=5, horizon=60),
            "G2_180_10_60": dict(context=90, gap=5, horizon=30),
            "G3_180_10_120": dict(context=90, gap=5, horizon=60),
            "G4_300_10_60": dict(context=150, gap=5, horizon=30)},
        development_devices=["01-X450-1", "02-X450-2", "03-X450-3"], external_device="04-X450-4",
        excluded_jobs=["01-X450-1_20260726T083710714006Z", "01-X450-1_20260727T115726163177Z",
                       "01-X450-1_20260730T202943865451Z"],
        field_lineage=lineage, seeds=[20260928, 20260929, 20260930],
        model_params=dict(d_model=128, n_heads=2, e_layers=3, d_ff=512, factor=3, dropout=0.05,
            crossformer_seg_len=10, crossformer_win_size=2, revin_affine=True, condition_hidden=64,
            learning_rate=1e-4, weight_decay=1e-5, batch_size=32),
        selection=dict(epochs=50, patience=8, validation_job_fraction=0.2, train_cap_per_device=0),
        provenance=source["provenance"],
        field_policy="All 90 retained; no constant/coverage screening; no future added covariates",
        gap_policy="skip 5 ticks after last input: first label +12s; last +70s/+130s",
        evaluation_policy="04 held out from fitting/early stopping; business comparison is exploratory after prior use")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=HERE.parents[1])
    p.add_argument("--output", type=Path, default=HERE / "contract.json")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output, build(a.root.resolve()))
    print("Wrote full 218-field contract; no training.")

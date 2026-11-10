"""Download one published ALL90 model and verify every file with SHA-256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORY = "ganmingzhang68-glitch/LightX_Crossformer_Time_forscasting"
DIRECTORY = "models/all90_20261110"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def download(group, seed, output, ref="experiment/all90-latency-4groups"):
    base = "https://raw.githubusercontent.com/" + REPOSITORY + "/" + quote(ref, safe="/") + "/" + DIRECTORY + "/"
    headers = {"User-Agent": "ALL90-model-downloader"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    def get(relative):
        with urlopen(Request(base + quote(relative, safe="/"), headers=headers), timeout=180) as response:
            return response.read()
    index = json.loads(get("MODEL_INDEX.json"))
    record = next((r for r in index["models"] if r["group"] == group and r["seed"] == seed), None)
    if record is None:
        raise ValueError("Unknown group/seed")
    manifest_bytes = get(record["path"] + "/bundle.json")
    manifest = json.loads(manifest_bytes)
    expected = manifest.pop("bundle_sha")
    if digest(manifest) != expected or expected != record["bundle_sha"]:
        raise ValueError("Model manifest checksum mismatch")
    manifest["bundle_sha"] = expected
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for relative, checksum in manifest["files"].items():
        path = (output / relative).resolve()
        if output not in path.parents:
            raise ValueError("Unsafe model file path")
        data = get(record["path"] + "/" + relative)
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError("File checksum mismatch: " + relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print("Verified", relative, flush=True)
    (output / "bundle.json").write_bytes(manifest_bytes)
    print("Complete:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("G1_300_10_120", "G2_180_10_60", "G3_180_10_120", "G4_300_10_60"), required=True)
    parser.add_argument("--seed", type=int, choices=(20260928, 20260929, 20260930), default=20260928)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref", default="experiment/all90-latency-4groups")
    args = parser.parse_args()
    download(args.group, args.seed, args.output, args.ref)

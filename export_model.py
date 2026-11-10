"""Export a trusted new matrix/ALL90 checkpoint to a standalone inference bundle."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import torch


ROOT = Path(__file__).resolve().parent


def export_model(model_dir, output, experiment="matrix"):
    if experiment == "matrix":
        from experiments.input_window_matrix_v1 import common, training, data, model
    elif experiment == "all90":
        from experiments.all90_latency_v1 import common, training, data, model
    else:
        raise ValueError("Unknown experiment")
    model_dir, output = Path(model_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    for protected in (model_dir, ROOT):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Export output must not overlap source models or repository")
    # Only export checkpoints produced by the matching, unchanged training code.
    trained, state, spec = training.load_model(model_dir, torch.device("cpu"))
    c = common.contract()
    group = c["groups"][state["group"]]
    if group["gap"] != 5 or c["cadence_seconds"] != 2:
        raise ValueError("Only models trained with a 10s gap can be exported")
    if experiment == "matrix" and state["contract"].get("storage_contract_sha") != common.digest(c):
        raise ValueError("Model and source field contracts differ")
    output.mkdir(parents=True)
    runtime = output / "runtime"
    runtime.mkdir()
    (runtime / "__init__.py").write_text('"""Frozen inference implementation."""\n', encoding="utf-8")
    model_source = Path(model.__file__).read_text(encoding="utf-8").replace("from .vendor.crossformer", "from .crossformer")
    (runtime / "model.py").write_text(model_source, encoding="utf-8")
    shutil.copyfile(common.HERE / "vendor/crossformer.py", runtime / "crossformer.py")
    shutil.copyfile(common.HERE / "vendor/LICENSE", runtime / "LICENSE")
    # Freeze the exact training encoder without copying dataset/training dependencies.
    tree = ast.parse(Path(data.__file__).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ("Encoder", "event_channels")]
    encoder_source = "import hashlib\nimport json\nimport numpy as np\n\ndef digest(value):\n    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()\n\n"
    encoder_source += "\n\n".join(ast.unparse(n) for n in nodes) + "\n"
    (runtime / "encoding.py").write_text(encoder_source, encoding="utf-8")
    shutil.copyfile(ROOT / "portable_inference.py", output / "predict.py")
    shutil.copyfile(ROOT / "forecast_windows.py", output / "forecast_windows.py")
    torch.save(trained.state_dict(), output / "weights.pt")
    metadata = dict(version=1, experiment=experiment, storage_contract=c, model_contract=state["contract"],
        group=state["group"], preprocessor=state["pre"], params=state["params"],
        forecast_gap_seconds=10, source_code_sha=state["code_sha"], source_signature=spec["signature"],
        source_checkpoint_sha=common.file_hash(model_dir / "best.pt"), epoch=state["epoch"],
        input_scope="aligned, numeric/type-decoded historical fields; no raw-source fusion during inference")
    metadata["files"] = {p.relative_to(output).as_posix(): common.file_hash(p)
                         for p in sorted(output.rglob("*")) if p.is_file()}
    metadata["bundle_sha"] = common.digest(metadata)
    common.write_json(output / "bundle.json", metadata)
    (output / "requirements.txt").write_text("numpy>=1.26\npandas>=2.2\npyarrow>=16\ntorch>=2.4\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Standalone 10s-gap inference bundle\n\n"
        "Install requirements.txt. Input must be an aligned numeric wide CSV/Parquet with UTC timestamps, "
        "original128 and all selected input fields. Date fields must already use the training log1p age encoding. "
        "Quality/event models require observed_mask, new_observed, new_age_seconds. "
        "Keep device_id, job_id and segment_id to preserve boundaries.\n\n"
        "python predict.py --input aligned.parquet --output predictions.parquet --mode latest --device cpu\n\n"
        "Rolling: --mode rolling --include-target. First label is +12s after the last input. "
        "Do not edit bundle.json, weights.pt or runtime files; retain all files when transferring the bundle.\n",
        encoding="utf-8")
    # Verify portable reconstruction numerically against the trusted source implementation.
    from portable_inference import PortableModel
    portable = PortableModel(output, "cpu")
    import numpy as np
    raw = np.tile(np.asarray(state["pre"]["mean"], dtype="float32"), (group["context"], 1))
    quality = np.ones((group["context"], 3), dtype="float32")
    encoded = portable.encoder.encode(raw, quality)
    with torch.inference_mode():
        expected = trained(*(torch.from_numpy(v.copy()).unsqueeze(0) for v in encoded)).numpy()
        actual = portable.model(*(torch.from_numpy(v.copy()).unsqueeze(0) for v in encoded)).numpy()
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Nonfinite export verification predictions")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    return dict(output=str(output), bundle_sha=metadata["bundle_sha"], group=state["group"],
                outputs=len(c["targets"]), forecast_gap_seconds=10, numerical_roundtrip=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=("matrix", "all90"), default="matrix")
    args = parser.parse_args()
    print(json.dumps(export_model(args.model_dir, args.output, args.experiment), ensure_ascii=False, indent=2))

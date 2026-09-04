from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from crossformer import Crossformer, CrossformerConfig


def main() -> None:
    root = Path(__file__).resolve().parent
    raw_config = json.loads(
        (root / "artifacts/model_config.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "artifacts/feature_manifest.json").read_text(encoding="utf-8")
    )
    scaler = json.loads(
        (root / "artifacts/scaler.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (root / "artifacts/final_metrics.json").read_text(encoding="utf-8")
    )
    history = json.loads(
        (root / "artifacts/training_history.json").read_text(encoding="utf-8")
    )
    config = CrossformerConfig.from_dict(raw_config)
    features = list(manifest["feature_order"])
    boolean = list(manifest["boolean_features"])
    continuous = list(manifest["continuous_features"])
    if len(features) != config.enc_in or len(features) != len(set(features)):
        raise RuntimeError("Feature count/order is inconsistent with model configuration")
    if set(boolean) & set(continuous) or set(boolean) | set(continuous) != set(features):
        raise RuntimeError("Boolean and continuous feature groups are not an exact partition")
    if list(scaler["feature_order"]) != features:
        raise RuntimeError("Scaler feature order differs from the model feature contract")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    if mean.shape != (config.enc_in,) or scale.shape != (config.enc_in,):
        raise RuntimeError("Scaler vector shape differs from model feature count")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("Scaler contains invalid mean/scale values")
    best = min(history, key=lambda row: float(row["validation_loss"]))
    if int(best["epoch"]) != int(raw_config["best_epoch"]):
        raise RuntimeError("Training history and model config disagree on best epoch")
    if int(best["epoch"]) != int(metrics["best_epoch_on_training_device_temporal_validation"]):
        raise RuntimeError("Training history and final metrics disagree on best epoch")
    checkpoint = root / "artifacts/checkpoint.pth"
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256 != raw_config["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA-256 differs from model configuration")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model = Crossformer(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, config.seq_len, config.enc_in))
    if output.shape != (2, config.pred_len, config.enc_in):
        raise RuntimeError(f"Unexpected output shape: {list(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Model produced NaN or infinite values")
    print(json.dumps({
        "strict_checkpoint_load": True,
        "artifact_contract_valid": True,
        "feature_count": len(features),
        "continuous_feature_count": len(continuous),
        "boolean_feature_count": len(boolean),
        "scaler_training_rows": int(scaler["n_samples_seen"]),
        "best_epoch": int(best["epoch"]),
        "input_shape": [2, config.seq_len, config.enc_in],
        "output_shape": list(output.shape),
        "checkpoint_sha256": checkpoint_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from crossformer import Crossformer, CrossformerConfig


def main() -> None:
    root = Path(__file__).resolve().parent
    config = CrossformerConfig.from_dict(
        json.loads((root / "artifacts/model_config.json").read_text(encoding="utf-8"))
    )
    checkpoint = root / "artifacts/checkpoint.pth"
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
        "input_shape": [2, config.seq_len, config.enc_in],
        "output_shape": list(output.shape),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()

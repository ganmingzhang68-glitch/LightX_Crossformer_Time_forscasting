"""Explicit CUDA-only training commands; importing this file never trains."""
from __future__ import annotations

import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import code_identity, digest, read_json, write_json, validate_selection
from model import ForecastModel, grouped_loss
from metrics import evaluate
from amp_step import safe_optimizer_step


def loader(ds, batch_size=32, shuffle=False, seed=0, workers=0):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        generator=torch.Generator().manual_seed(seed), pin_memory=torch.cuda.is_available())


def save_checkpoint(path, state):
    temp = Path(path).with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def train(train_data, val_data, c, group, pre, params, output, epochs, patience, seed,
          precision="fp32", workers=0, resume=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; local verification is forward-only")
    if train_data.repo.manifest["external_only"] or (val_data and val_data.repo.manifest["external_only"]):
        raise ValueError("04 cannot enter training or early stopping")
    validate_selection(pre["fit_selection"], c)
    if epochs < 1 or patience < 1 or precision not in ("fp32", "fp16"):
        raise ValueError("Invalid training budget/precision")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    out = Path(output)
    spec = dict(code_sha=code_identity(), contract_sha=digest(c), data_sha=digest(train_data.repo.manifest),
        group=group, preprocessor=pre, params=params, epochs=epochs, patience=patience, seed=seed,
        precision=precision, train_items_sha=digest(train_data.items),
        validation_items_sha=digest(val_data.items) if val_data else None)
    signature = digest(spec)
    if out.exists():
        if not resume or read_json(out / "spec.json")["signature"] != signature:
            raise ValueError("Existing output or different resume contract")
        if (out / "result.json").exists():
            return read_json(out / "result.json")
    else:
        out.mkdir(parents=True)
        write_json(out / "spec.json", dict(signature=signature, **spec))
    model = ForecastModel(c, group, pre, params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    start, stale, best_epoch, best, history = 0, 0, 0, float("inf"), []
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Resume checkpoint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["amp"])
        start, stale, best_epoch, best, history = (state[k] for k in ("epoch", "stale", "best_epoch", "best", "history"))
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start + 1, epochs + 1):
        if val_data and stale >= patience:
            break
        tick = time.perf_counter()
        model.train()
        total, count, retries, fallbacks = 0.0, 0, 0, 0
        for dynamic, conditions, aux, target, _ in loader(train_data, params["batch_size"], True, seed + epoch, workers):
            dynamic, conditions, aux, target = (v.to(device) for v in (dynamic, conditions, aux, target))
            loss, telemetry = safe_optimizer_step(model, optimizer, scaler,
                lambda: grouped_loss(model(dynamic, conditions, aux), target, c), device)
            total += float(loss) * len(target)
            count += len(target)
            retries += telemetry["amp_retries"]
            fallbacks += int(telemetry["fp32_fallback"])
        scheduler.step()
        validation = evaluate(model, loader(val_data, params["batch_size"], workers=workers), c, pre, device,
                              precision=precision, detailed=False) if val_data else None
        score = validation["objective"] if validation else None
        if validation and (score is None or not np.isfinite(score)):
            raise ValueError("Validation objective is unscorable")
        if not val_data or score < best:
            best, best_epoch, stale = score if score is not None else 0.0, epoch, 0
            save_checkpoint(out / "best.pt", dict(model=model.state_dict(), contract=c, group=group, pre=pre,
                params=params, epoch=epoch, seed=seed, signature=signature, code_sha=code_identity()))
        else:
            stale += 1
        torch.cuda.synchronize()
        history.append(dict(epoch=epoch, train_loss=total/count, validation_objective=score,
            epoch_seconds=time.perf_counter()-tick, amp_retries=retries, fp32_fallback_batches=fallbacks,
            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        save_checkpoint(out / "last.pt", dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), amp=scaler.state_dict(), epoch=epoch, stale=stale, best=best,
            best_epoch=best_epoch, history=history, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        write_json(out / "history.json", history)
        print(group, seed, history[-1], flush=True)
    if not best_epoch:
        raise RuntimeError("No completed checkpoint")
    result = dict(group=group, seed=seed, best_epoch=best_epoch, best_score=best if val_data else None,
        epochs_completed=len(history), signature=signature, train_windows=train_data.counts,
        validation_windows=val_data.counts if val_data else {}, raw_input_fields=218, added_fields=90,
        training_seconds=sum(r["epoch_seconds"] for r in history), precision=precision,
        parameter_count=sum(p.numel() for p in model.parameters()))
    write_json(out / "result.json", result)
    return result


def load_model(model_dir, device):
    # Only load trusted checkpoints generated by this experiment.
    model_dir = Path(model_dir)
    state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    spec = read_json(model_dir / "spec.json")
    result = read_json(model_dir / "result.json")
    expected = digest({k: v for k, v in spec.items() if k != "signature"})
    if state["signature"] != expected or result["signature"] != expected or spec["signature"] != expected:
        raise ValueError("Model/spec/result identity mismatch")
    if state["code_sha"] != code_identity() or digest(state["contract"]) != spec["contract_sha"]:
        raise ValueError("Model code/contract mismatch")
    if state["pre"] != spec["preprocessor"] or state["params"] != spec["params"] or state["group"] != spec["group"]:
        raise ValueError("Checkpoint metadata mismatch")
    model = ForecastModel(state["contract"], state["group"], state["pre"], state["params"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state, spec

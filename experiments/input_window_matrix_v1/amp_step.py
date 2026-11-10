"""Bounded AMP recovery: retry the same batch/RNG before any optimizer update."""
from __future__ import annotations
import torch


def safe_optimizer_step(model, optimizer, scaler, loss_fn, device, max_amp_retries=4):
    """Return detached loss and telemetry. loss_fn performs forward + scalar loss.

    Each failed AMP backward is discarded. Lower the scale and replay the batch
    with identical Torch RNG (including dropout); never skip a training example.
    Persistent AMP failures get one FP32 attempt. Invalid FP32 loss/gradients
    remain fatal. Models in this experiment have no running BatchNorm buffers.
    """
    if max_amp_retries < 0:
        raise ValueError('max_amp_retries must be nonnegative')
    device = torch.device(device)
    amp = scaler.is_enabled()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    initial_scale = float(scaler.get_scale())
    events = []

    def restore_rng():
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    def finite_grads():
        flags = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        return bool(torch.stack(flags).all()) if flags else False

    # The final attempt is FP32, even when AMP is enabled.
    for attempt in range(max_amp_retries + 2 if amp else 1):
        use_amp = amp and attempt <= max_amp_retries
        if attempt:
            restore_rng()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = loss_fn()
        if not bool(torch.isfinite(loss)):
            if not use_amp:
                raise FloatingPointError('Nonfinite FP32 loss; no optimizer update')
            events.append({'attempt': attempt, 'reason': 'nonfinite_amp_loss',
                           'scale': float(scaler.get_scale())})
            # Scaling cannot fix forward overflow. Go directly to FP32 below.
            break
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if finite_grads():
            # Keep strict norm validation; do not normalize nonfinite gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            return loss.detach(), {'amp_retries': len(events), 'fp32_fallback': amp and not use_amp,
                'initial_scale': initial_scale, 'final_scale': float(scaler.get_scale()), 'events': events}
        if not use_amp:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite FP32 gradients; no optimizer update')
        old_scale = float(scaler.get_scale())
        # update(new_scale) clears GradScaler's UNSCALED stage without calling
        # optimizer.step on invalid gradients. The next replay starts cleanly.
        scaler.update(new_scale=old_scale * scaler.get_backoff_factor())
        events.append({'attempt': attempt, 'reason': 'nonfinite_amp_gradients',
                       'scale': old_scale, 'next_scale': float(scaler.get_scale())})
    # Forward overflow reached the break above. Retry exactly once in FP32.
    restore_rng()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=False):
        loss = loss_fn()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite FP32 loss; no optimizer update')
    loss.backward()
    if not finite_grads():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError('Nonfinite FP32 gradients; no optimizer update')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), {'amp_retries': len(events), 'fp32_fallback': True,
        'initial_scale': initial_scale, 'final_scale': float(scaler.get_scale()), 'events': events}

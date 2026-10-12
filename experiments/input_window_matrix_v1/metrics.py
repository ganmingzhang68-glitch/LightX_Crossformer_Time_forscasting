"""Full/common-60s quality, per-feature/task/lead metrics and persistence."""
from __future__ import annotations

import numpy as np
import torch


def ratio(a, b):
    return float(a / b) if b > 1e-12 else None


class Accumulator:
    def __init__(self, c, pre):
        self.c, self.pre, self.n = c, pre, 0
        self.width = len(c["targets"])
        for key in ("sse", "bse", "sae", "bae", "tp", "fp", "fn", "btp", "bfp", "bfn", "pos", "etp", "efp", "efn", "betp", "befp", "befn"):
            setattr(self, key, np.zeros(self.width, dtype="float64"))

    def add(self, pred, truth, baseline):
        if pred.shape != truth.shape or baseline.shape != truth.shape or pred.shape[-1] != self.width:
            raise ValueError("Invalid evaluation shape")
        if not all(np.isfinite(x).all() for x in (pred, truth, baseline)):
            raise ValueError("Nonfinite evaluation values")
        ps, ys, bs = pred >= 0.5, truth >= 0.5, baseline >= 0.5
        pe, ye, be = ps[:, 1:] != ps[:, :-1], ys[:, 1:] != ys[:, :-1], bs[:, 1:] != bs[:, :-1]
        for prefix, event in (("e", pe), ("be", be)):
            getattr(self, prefix + "tp")[:] += (event & ye).sum((0, 1))
            getattr(self, prefix + "fp")[:] += (event & ~ye).sum((0, 1))
            getattr(self, prefix + "fn")[:] += (~event & ye).sum((0, 1))
        self.n += truth.shape[0] * truth.shape[1]
        self.sse += ((pred - truth) ** 2).sum((0, 1))
        self.bse += ((baseline - truth) ** 2).sum((0, 1))
        self.sae += np.abs(pred - truth).sum((0, 1))
        self.bae += np.abs(baseline - truth).sum((0, 1))
        self.pos += ys.sum((0, 1))
        for prefix, prediction in (("", ps), ("b", bs)):
            getattr(self, prefix + "tp")[:] += (prediction & ys).sum((0, 1))
            getattr(self, prefix + "fp")[:] += (prediction & ~ys).sum((0, 1))
            getattr(self, prefix + "fn")[:] += (~prediction & ys).sum((0, 1))

    def finish(self):
        if not self.n:
            raise ValueError("Empty evaluation")
        active = self.pre["active_source_fields"]
        score = self.c["score_targets"]
        ci = [self.c["targets"].index(f) for f in self.c["continuous"] if f in active and f in score]
        bi = [self.c["targets"].index(f) for f in self.c["boolean"] if f in active and f in score]
        scale = np.asarray(self.pre["scale"])[[self.c["source_features"].index(f) for f in self.c["targets"]]]
        f1 = np.divide(2 * self.tp, 2 * self.tp + self.fp + self.fn, out=np.zeros(self.width), where=(2 * self.tp + self.fp + self.fn) > 0)
        bf1 = np.divide(2 * self.btp, 2 * self.btp + self.bfp + self.bfn, out=np.zeros(self.width), where=(2 * self.btp + self.bfp + self.bfn) > 0)
        relative = ratio((self.sse[ci] / scale[ci] ** 2).sum(), (self.bse[ci] / scale[ci] ** 2).sum())
        macro = float(f1[bi].mean()) if bi else None
        objective = 0.7 * relative + 0.3 * (1 - macro) if relative is not None and macro is not None else None
        features = {}
        for i, field in enumerate(self.c["targets"]):
            r = ratio(self.sse[i], self.bse[i])
            row = dict(mae=float(self.sae[i] / self.n), mse=float(self.sse[i] / self.n),
                rmse=float(np.sqrt(self.sse[i] / self.n)), persistence_mae=float(self.bae[i] / self.n),
                persistence_mse=float(self.bse[i] / self.n), relative_mse=r,
                skill=None if r is None else 1 - r, active_in_training=field in active)
            if field in self.c["boolean"]:
                row.update(f1=float(f1[i]), persistence_f1=float(bf1[i]), positives=int(self.pos[i]),
                    precision=ratio(self.tp[i], self.tp[i] + self.fp[i]), recall=ratio(self.tp[i], self.tp[i] + self.fn[i]),
                    accuracy=float(1 - (self.fp[i] + self.fn[i]) / self.n),
                    false_positive_rate=ratio(self.fp[i], self.n - self.pos[i]),
                    transition_f1=ratio(2*self.etp[i], 2*self.etp[i]+self.efp[i]+self.efn[i]),
                    transition_recall=ratio(self.etp[i], self.etp[i]+self.efn[i]),
                    persistence_transition_f1=ratio(2*self.betp[i], 2*self.betp[i]+self.befp[i]+self.befn[i]))
            features[field] = row
        return dict(points=self.n, objective=objective, continuous_relative_mse=relative,
            continuous_skill=None if relative is None else 1-relative,
            boolean_macro_f1=macro, persistence_boolean_macro_f1=float(bf1[bi].mean()) if bi else None,
            active_continuous=len(ci), active_boolean=len(bi), by_feature=features)


def evaluate(model, loader, c, pre, device, precision="fp32", detailed=True):
    from contextlib import nullcontext
    from common import catalog
    target_indices = [c["source_features"].index(f) for f in c["targets"]]
    mean, scale = np.asarray(pre["mean"])[target_indices], np.asarray(pre["scale"])[target_indices]
    target_positions = [c["profile"]["dynamic"].index(f) for f in c["targets"]]
    full, shared = Accumulator(c, pre), Accumulator(c, pre)
    jobs, shared_jobs, horizons = {}, {}, {}
    target27 = catalog()["profiles"]["A128_27"]["targets"]
    common_positions = [c["targets"].index(f) for f in target27]
    common_c = dict(c, targets=target27, score_targets=target27,
        continuous=[f for f in target27 if f in c["continuous"]], boolean=[f for f in target27 if f in c["boolean"]])
    common27 = Accumulator(common_c, pre)
    common27_jobs = {}
    model.eval()
    with torch.inference_mode():
        for dynamic, conditions, aux, target, keys in loader:
            context = torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()
            with context:
                output = model(*(v.to(device) for v in (dynamic, conditions, aux)))
            prediction = output.float().cpu().numpy().astype("float64") * scale + mean
            truth = target.numpy().astype("float64") * scale + mean
            last = dynamic[:, -1:, target_positions].numpy().astype("float64") * scale + mean
            baseline = np.broadcast_to(last, truth.shape)
            full.add(prediction, truth, baseline)
            shared.add(prediction[:, :30], truth[:, :30], baseline[:, :30])
            cp, cy, cb = (v[:, :30, common_positions] for v in (prediction, truth, baseline))
            common27.add(cp, cy, cb)
            if detailed:
                for key in set(keys):
                    select = np.array([x == key for x in keys])
                    jobs.setdefault(key, Accumulator(c, pre)).add(prediction[select], truth[select], baseline[select])
                    shared_jobs.setdefault(key, Accumulator(c, pre)).add(prediction[select, :30], truth[select, :30], baseline[select, :30])
                    common27_jobs.setdefault(key, Accumulator(common_c, pre)).add(cp[select], cy[select], cb[select])
                for start in range(0, truth.shape[1], 15):
                    stop = min(start + 15, truth.shape[1])
                    label = f"{12+start*2}-{12+(stop-1)*2}s"
                    horizons.setdefault(label, Accumulator(c, pre)).add(prediction[:, start:stop], truth[:, start:stop], baseline[:, start:stop])
    result = full.finish()
    result["common_first_60s"] = shared.finish()
    result["shared27_common_first_60s"] = common27.finish()
    result["scoring_targets"] = c["score_targets"]
    result["shared27_targets"] = target27
    if detailed:
        result.update(by_job={k: v.finish() for k, v in jobs.items()},
            common60_by_job={k: v.finish() for k, v in shared_jobs.items()},
            shared27_common60_by_job={k: v.finish() for k, v in common27_jobs.items()},
            by_lead={k: v.finish() for k, v in horizons.items()},
            notes="Original units; boolean threshold 0.5; active sets fixed on training; transition metrics exclude the unobserved gap boundary.")
    return result

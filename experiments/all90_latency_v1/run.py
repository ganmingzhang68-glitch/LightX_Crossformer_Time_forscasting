"""Explicit plan/preflight/select/refit/evaluate04 commands. No automatic training."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from common import contract, code_identity, digest, file_hash, read_json, safe_new_output, write_json


def plan(args):
    c = contract(args.contract)
    print("218 source fields = original128 + ALL90; outputs original128. No training.")
    for name, g in c["groups"].items():
        print(name, f"points={g['context']}/{g['gap']}/{g['horizon']}",
              f"first_label=+{2*(g['gap']+1)}s", f"last_label=+{2*(g['gap']+g['horizon'])}s")
    print("Three paired seeds per group; 01/02/03 chronological whole-job selection; 04 quality/timing only.")


def preflight(args):
    from build_contract import build
    from prepare import verify_schema
    c = contract(args.contract)
    actual = build(args.root.resolve())
    if actual != c:
        raise ValueError("Docs/model feature sources have changed; rebuild into a new experiment version")
    verify_schema(args.root / "data", c, c["development_devices"])
    old = read_json(args.root / "data/processed_crossformer_multidevice_july_5m_10s_128features/feature_manifest.json")
    if old["feature_order"] != c["targets"] or old.get("continuous_values_clipped", False):
        raise ValueError("Old prepared source contract differs")
    result = dict(status="passed", source_fields=218, added_fields=90, output_fields=128,
                  contract_sha=digest(c), code_sha=code_identity(), groups=c["groups"],
                  training_started=False, device04_values_read=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    print(result)


def reserve_run(output, budget, protected, resume):
    output = Path(output).resolve()
    if output.exists():
        if not resume or read_json(output / "run.json") != budget:
            raise ValueError("Existing run or different frozen budget")
    else:
        safe_new_output(output, protected).mkdir(parents=True)
        write_json(output / "run.json", budget)
    return output


def select(args):
    from data import Repository, Windows, split_jobs, fit_preprocessor
    from training import train
    c = contract(args.contract)
    repo = Repository(args.data_root, c)
    split = split_jobs(repo)
    pre = fit_preprocessor(repo, split["train"])
    budget = dict(contract_sha=digest(c), code_sha=code_identity(), data_sha=digest(repo.manifest),
                  precision=args.precision, selection=c["selection"], groups=c["groups"], seeds=c["seeds"])
    out = reserve_run(args.output, budget, [args.data_root], args.resume)
    write_json(out / "split.json", split)
    results = {}
    for group in c["groups"]:
        training = Windows(repo, split["train"], group, pre, balance=True,
                           cap=c["selection"]["train_cap_per_device"])
        validation = Windows(repo, split["validation"], group, pre, balance=False)
        runs = []
        for seed in c["seeds"]:
            runs.append(train(training, validation, c, group, pre, c["model_params"], out / group / f"seed_{seed}",
                c["selection"]["epochs"], c["selection"]["patience"], seed, args.precision, args.workers, args.resume))
        results[group] = dict(runs=runs, fixed_epochs=max(1, int(statistics.median(r["best_epoch"] for r in runs))))
    frozen = dict(**budget, split=split, results=results, params=c["model_params"],
        selection_rule="fixed equal architecture/hyperparameters; each group uses median best epoch across the 3 paired seeds; no 04 selection")
    frozen["freeze_sha"] = digest(frozen)
    write_json(out / "frozen.json", frozen)
    print("Selection complete; final training requires a separate refit command.")


def verify_freeze(frozen, c, repo=None):
    if digest({k: v for k, v in frozen.items() if k != "freeze_sha"}) != frozen["freeze_sha"]:
        raise ValueError("Frozen selection checksum mismatch")
    if frozen["contract_sha"] != digest(c) or frozen["code_sha"] != code_identity():
        raise ValueError("Frozen contract/code differs")
    if repo is not None and frozen["data_sha"] != digest(repo.manifest):
        raise ValueError("Frozen development data differs")


def refit(args):
    from data import Repository, Windows, fit_preprocessor
    from training import train
    c = contract(args.contract)
    repo = Repository(args.data_root, c)
    frozen = read_json(args.frozen)
    verify_freeze(frozen, c, repo)
    pre = fit_preprocessor(repo, frozen["split"]["refit"])
    budget = dict(freeze_sha=frozen["freeze_sha"], data_sha=digest(repo.manifest), code_sha=code_identity())
    out = reserve_run(args.output, budget, [args.data_root, args.frozen.parent], args.resume)
    results = []
    for group in c["groups"]:
        ds = Windows(repo, frozen["split"]["refit"], group, pre, balance=True)
        for seed in c["seeds"]:
            dest = out / group / f"seed_{seed}"
            result = train(ds, None, c, group, pre, frozen["params"], dest,
                frozen["results"][group]["fixed_epochs"], 1, seed, frozen["precision"], args.workers, args.resume)
            results.append(dict(path=dest.relative_to(out).as_posix(), group=group, seed=seed,
                                checkpoint_sha=file_hash(dest / "best.pt"), **{k: result[k] for k in ("best_epoch", "parameter_count")}))
    write_json(out / "final.json", dict(status="complete", frozen=frozen, models=results,
        data_policy={k: repo.manifest[k] for k in ("stride", "added_delay_seconds", "cohort", "date_encoding", "fill")},
        development_data_sha=digest(repo.manifest), all_seeds_reported=True))
    print("Final training complete; external evaluation/timing requires separate commands.")


def final_models(model_root, c, repo, device):
    from training import load_model
    final = read_json(model_root / "final.json")
    if final["status"] != "complete":
        raise ValueError("Final runs are incomplete")
    verify_freeze(final["frozen"], c)
    if {k: repo.manifest[k] for k in final["data_policy"]} != final["data_policy"]:
        raise ValueError("External cohort/availability policy differs from training")
    expected = {(g, seed) for g in c["groups"] for seed in c["seeds"]}
    if len(final["models"]) != len(expected) or {(r["group"], r["seed"]) for r in final["models"]} != expected:
        raise ValueError("All groups and all paired seeds must be present")
    result = []
    for record in final["models"]:
        path = (model_root / record["path"]).resolve()
        if model_root.resolve() not in path.parents:
            raise ValueError("Model path escapes experiment")
        if file_hash(path / "best.pt") != record["checkpoint_sha"]:
            raise ValueError("Checkpoint changed")
        model, state, spec = load_model(path, device)
        group, seed = record["group"], record["seed"]
        if (state["group"], state["seed"]) != (group, seed) or state["epoch"] != final["frozen"]["results"][group]["fixed_epochs"]:
            raise ValueError("Model group/seed/epoch differs from frozen result")
        if spec["validation_items_sha"] is not None or state["pre"]["fit_selection"] != final["frozen"]["split"]["refit"]:
            raise ValueError("Expected a final fixed-epoch refit, not a selection checkpoint")
        if spec["data_sha"] != final["development_data_sha"] or digest(state["contract"]) != digest(c):
            raise ValueError("Final checkpoint source identity differs")
        if state["params"] != final["frozen"]["params"] or spec["precision"] != final["frozen"]["precision"]:
            raise ValueError("Final model differs from frozen parameters/precision")
        if spec["seed"] != seed or spec["epochs"] != state["epoch"]:
            raise ValueError("Final model differs from fixed seed/epoch budget")
        result.append((record, model, state))
    return final, result


def evaluate04(args):
    import torch
    from data import Repository, Windows
    from training import loader
    from metrics import evaluate
    from compare import quality_summary
    c = contract(args.contract)
    repo = Repository(args.data_root, c, external=True)
    device = torch.device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 timing/quality requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    final, models = final_models(args.model_root, c, repo, device)
    out = safe_new_output(args.output, [args.data_root, args.model_root])
    out.mkdir(parents=True)
    selection = {c["external_device"]: repo.jobs(c["external_device"])}
    records = []
    for record, model, state in models:
        ds = Windows(repo, selection, record["group"], state["pre"])
        metrics = evaluate(model, loader(ds, args.batch_size, workers=args.workers), c, state["pre"], device, args.precision)
        metrics.update(group=record["group"], seed=record["seed"], precision=args.precision,
            horizon_points=c["groups"][record["group"]]["horizon"],
            checkpoint_sha=record["checkpoint_sha"], window_selection_sha=digest(ds.items),
            active_targets=state["pre"]["active_targets"], data_sha=digest(repo.manifest),
            external_exploratory=True, selected_windows=ds.counts)
        name = f"{record['group']}_seed_{record['seed']}.json"
        write_json(out / name, metrics)
        records.append(metrics)
    write_json(out / "quality_summary.json", quality_summary(records))
    write_json(out / "evaluation.json", dict(status="complete", freeze_sha=final["frozen"]["freeze_sha"],
        models=[dict(group=r["group"], seed=r["seed"], checkpoint_sha=r["checkpoint_sha"]) for r in records],
        data_sha=digest(repo.manifest), precision=args.precision, device=str(device), all_seeds_reported=True))
    print("Wrote full/common60 quality and paired task bootstrap summaries:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("plan")
    q.set_defaults(func=plan)
    q = sub.add_parser("preflight")
    q.add_argument("--root", type=Path, required=True)
    q.add_argument("--output", type=Path)
    q.set_defaults(func=preflight)
    s = sub.add_parser("select")
    s.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    s.set_defaults(func=select)
    r = sub.add_parser("refit")
    r.add_argument("--frozen", type=Path, required=True)
    r.set_defaults(func=refit)
    e = sub.add_parser("evaluate04")
    e.add_argument("--model-root", type=Path, required=True)
    e.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    e.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    e.add_argument("--batch-size", type=int, default=32)
    e.set_defaults(func=evaluate04)
    for q in (s, r, e):
        q.add_argument("--data-root", type=Path, required=True)
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--workers", type=int, default=0)
    for q in (s, r):
        q.add_argument("--resume", action="store_true")
    for q in sub.choices.values():
        q.add_argument("--contract", type=Path)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

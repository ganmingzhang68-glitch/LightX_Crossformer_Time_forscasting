"""plan/init are CPU-only; explicit run trains ONE model at a time until completion."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import gc
import os
from pathlib import Path
import statistics
import sys
import traceback
from datetime import datetime, timezone

from common import (HERE, contract, catalog, code_identity, digest, file_hash, read_json,
                    write_json, safe_new_output, selected_profiles, load_feature_manifests,
                    runtime_contract, verify_signed)


def now():
    return datetime.now(timezone.utc).isoformat()


def make_plan(profiles=None, output_policy="old128", seeds=None, epochs=50, patience=8,
              train_cap=0, workers=0, feature_manifests=None, custom_profiles=None,
              custom_sources=None):
    if output_policy != "old128":
        raise ValueError("This new training matrix must output the original128 fields")
    if min(epochs, patience) < 1 or train_cap < 0 or workers < 0:
        raise ValueError("Invalid budget")
    if feature_manifests is not None and (custom_profiles is not None or custom_sources is not None):
        raise ValueError("Use manifest paths or frozen custom profiles, not both")
    c = contract()
    if feature_manifests is not None:
        custom_profiles, custom_sources = load_feature_manifests(feature_manifests)
    else:
        custom_profiles, custom_sources = copy.deepcopy(custom_profiles or {}), copy.deepcopy(custom_sources or [])
    chosen, aliases = selected_profiles(profiles, output_policy, custom_profiles)
    seeds = seeds if seeds is not None else c["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Require unique nonnegative seeds")
    tasks = []
    # Freeze every pair on development data before beginning any final fit.
    for stage in ("select", "refit"):
        for profile in chosen:
            for group in c["groups"]:
                for seed in seeds:
                    key = f"{profile}/{group}/seed_{seed}"
                    tasks.append(dict(id=f"{stage}/{key}", stage=stage, profile=profile, group=group, seed=seed,
                                      path=f"{'selection' if stage == 'select' else 'models'}/{key}"))
    result = dict(version=2, profiles=chosen, requested_profiles=list(profiles) if profiles else None,
        custom_profiles=custom_profiles, feature_manifest_sources=custom_sources,
        output_policy=output_policy, aliases=aliases, groups=c["groups"], seeds=seeds,
        params=c["model_params"], selection=dict(epochs=epochs, patience=patience,
        train_cap_per_device=train_cap, validation_job_fraction=c["selection"]["validation_job_fraction"]),
        workers=workers, precision="fp32", tasks=tasks, code_sha=code_identity(),
        storage_contract_sha=digest(c), catalog_sha=digest(catalog()),
        training_policy="paired seeds; per-input/window median best epoch; fixed development refit; 04 never selects")
    return result


def describe(plan):
    print("No training started. Historical input/output matrix:")
    for name, p in plan["profiles"].items():
        print(f"{name}: raw={p['raw_input_count']} dynamic={len(p['dynamic'])} conditions={len(p['conditions'])} "
              f"outputs={len(p['targets'])} quality={p['quality']} masks={p['missing_masks']} events={p['events']}")
    for name, g in plan["groups"].items():
        print(name, f"seconds={2*g['context']} / {2*g['gap']} / {2*g['horizon']}")
    models = len(plan["profiles"]) * len(plan["groups"]) * len(plan["seeds"])
    print(f"{len(plan['profiles'])} input/output contracts x 4 windows x {len(plan['seeds'])} seeds = {models} final models")
    print(f"{models} selection fits + {models} final fits = {len(plan['tasks'])} sequential training tasks")
    if plan["aliases"]:
        print("Identical converted contracts are deduplicated:", plan["aliases"])


def initialize(args):
    from data import Repository, split_jobs, fit_preprocessor, apply_shared_cohort
    plan = make_plan(args.profiles, args.output_policy, args.seeds, args.epochs, args.patience,
                     args.train_cap, args.workers, getattr(args, "feature_manifest", None))
    c = contract()
    repo = Repository(args.data_root, c)  # complete identity/time/observed-label validation
    cohort = apply_shared_cohort(repo, plan["profiles"])
    split = split_jobs(repo)
    train_pre = fit_preprocessor(repo, split["train"])
    refit_pre = fit_preprocessor(repo, split["refit"])
    for p in plan["profiles"].values():
        if any(train_pre["count"][c["source_features"].index(f)] == 0 for f in p["targets"]):
            raise ValueError("A supervised target is never observed in training")
    inputs = dict(split=split, train_pre=train_pre, refit_pre=refit_pre, cohort=cohort)
    plan.update(data_root=str(args.data_root.resolve()), data_sha=digest(repo.manifest), inputs_sha=digest(inputs))
    plan["data_policy"] = {k: repo.manifest[k] for k in ("stride", "added_delay_seconds", "cohort", "availability", "date_encoding", "fill")}
    plan["plan_sha"] = digest(plan)
    output = safe_new_output(args.output, [args.data_root, HERE.parent])
    output.mkdir(parents=True)
    write_json(output / "inputs.json", inputs)
    write_json(output / "plan.json", plan)
    write_json(output / "status.json", dict(state="ready", created_at=now(), started_at=None, plan_sha=plan["plan_sha"],
        completed_tasks=0, total_tasks=len(plan["tasks"]), current_task=None,
        tasks={t["id"]: dict(state="pending") for t in plan["tasks"]}))
    write_json(output / "preflight.json", dict(status="passed", training_started=False, device04_values_read=False,
        data_sha=plan["data_sha"], code_sha=plan["code_sha"], plan_sha=plan["plan_sha"],
        shared_windows=sum(cohort["windows_by_job"].values()), eligible_jobs=sum(map(len, split["refit"].values()))))
    describe(plan)
    print("Queue ready. Explicit train_queue.py run is required to start:", output)


def verify_plan(plan):
    verify_signed(plan, "plan_sha")
    if plan["code_sha"] != code_identity() or plan["storage_contract_sha"] != digest(contract()) or plan["catalog_sha"] != digest(catalog()):
        raise ValueError("Code or field contract changed; create a new queue/version")
    fresh = make_plan(plan["requested_profiles"], plan["output_policy"], plan["seeds"],
                      plan["selection"]["epochs"], plan["selection"]["patience"],
                      plan["selection"]["train_cap_per_device"], plan["workers"],
                      custom_profiles=plan["custom_profiles"], custom_sources=plan["feature_manifest_sources"])
    if any(plan[k] != v for k, v in fresh.items()):
        raise ValueError("Queue differs from reproducible task matrix")


@contextmanager
def exclusive_lock(output):
    """OS lock releases on process death; retained lock file isn't a stale lock."""
    stream = (Path(output) / ".queue.lock").open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    except OSError as exc:
        if not locked:
            raise RuntimeError("Another process owns this training queue") from exc
        raise
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def frozen_pair(output, plan, task, create=True):
    name, group = task["profile"], task["group"]
    results = [read_json(output / f"selection/{name}/{group}/seed_{seed}/result.json") for seed in plan["seeds"]]
    fixed = max(1, int(statistics.median(r["best_epoch"] for r in results)))
    result = dict(plan_sha=plan["plan_sha"], profile=name, group=group, fixed_epochs=fixed,
        selection_signatures=[r["signature"] for r in results],
        rule="median best epoch across the pre-registered paired seeds; development validation only")
    result["freeze_sha"] = digest(result)
    path = output / "frozen" / name / f"{group}.json"
    if path.exists():
        if read_json(path) != result:
            raise ValueError("Frozen epoch selection changed")
        return result
    if not create:
        raise ValueError("Final model is missing its frozen epoch selection")
    write_json(path, result)
    return result


def train_task(output, plan, inputs, repo, task):
    from data import Windows
    from training import train
    p = plan["profiles"][task["profile"]]
    c = runtime_contract(contract(), p)
    selection = task["stage"] == "select"
    pre = inputs["train_pre"] if selection else inputs["refit_pre"]
    ds = Windows(repo, inputs["split"]["train" if selection else "refit"], task["group"], pre, p,
                 balance=True, cap=plan["selection"]["train_cap_per_device"] if selection else 0)
    val = Windows(repo, inputs["split"]["validation"], task["group"], pre, p) if selection else None
    epochs = plan["selection"]["epochs"] if selection else frozen_pair(output, plan, task)["fixed_epochs"]
    dest = output / task["path"]
    return train(ds, val, c, task["group"], pre, plan["params"], dest,
        epochs, plan["selection"]["patience"] if selection else 1, task["seed"],
        precision=plan["precision"], workers=plan["workers"], resume=dest.exists())


def artifact_record(output, plan, inputs, task):
    dest = output / task["path"]
    spec, result = read_json(dest / "spec.json"), read_json(dest / "result.json")
    signature = digest({k: v for k, v in spec.items() if k != "signature"})
    selection = task["stage"] == "select"
    pre = inputs["train_pre"] if selection else inputs["refit_pre"]
    expected_epochs = plan["selection"]["epochs"] if selection else frozen_pair(output, plan, task, create=False)["fixed_epochs"]
    if (spec["signature"] != signature or result["signature"] != signature or
        spec["code_sha"] != plan["code_sha"] or spec["data_sha"] != plan["data_sha"] or
        spec["contract_sha"] != digest(runtime_contract(contract(), plan["profiles"][task["profile"]])) or
        spec["preprocessor"] != pre or spec["params"] != plan["params"] or spec["epochs"] != expected_epochs or
        spec["precision"] != plan["precision"] or spec["seed"] != task["seed"] or spec["group"] != task["group"] or
        (spec["validation_items_sha"] is not None) != selection or
        result["epochs_completed"] < 1 or not 1 <= result["best_epoch"] <= result["epochs_completed"]):
        raise ValueError("Training artifacts don't match the frozen queue task")
    if not selection and (result["best_epoch"] != expected_epochs or result["epochs_completed"] != expected_epochs):
        raise ValueError("Final model didn't complete its fixed-epoch budget")
    import torch
    checkpoint = torch.load(dest / "best.pt", map_location="cpu", weights_only=False)
    if (checkpoint["signature"] != signature or checkpoint["code_sha"] != plan["code_sha"] or
        checkpoint["pre"] != pre or checkpoint["params"] != plan["params"] or
        checkpoint["group"] != task["group"] or checkpoint["seed"] != task["seed"] or
        checkpoint["epoch"] != result["best_epoch"] or digest(checkpoint["contract"]) != spec["contract_sha"]):
        raise ValueError("Checkpoint metadata doesn't match the finished task")
    hashes = {name: file_hash(dest / name) for name in ("best.pt", "last.pt", "history.json", "spec.json", "result.json")}
    return dict(signature=signature, hashes=hashes, best_epoch=result["best_epoch"],
        parameter_count=result["parameter_count"], training_seconds=result["training_seconds"])


def execute_queue(output, plan, inputs, repo, resume=False, task_runner=None, verifier=None):
    """Synchronous loop. Test doubles verify orchestration without optimizer updates."""
    task_runner = task_runner or train_task
    verifier = verifier or artifact_record
    state = read_json(output / "status.json")
    if state["plan_sha"] != plan["plan_sha"] or set(state["tasks"]) != {t["id"] for t in plan["tasks"]}:
        raise ValueError("Status doesn't belong to this plan")
    if state["started_at"] and not resume:
        raise ValueError("Existing run requires --resume")
    state.update(state="running", started_at=state["started_at"] or now(), pid=os.getpid())
    write_json(output / "status.json", state)
    for task in plan["tasks"]:
        entry = state["tasks"][task["id"]]
        try:
            if entry["state"] == "complete":
                if verifier(output, plan, inputs, task) != entry["artifact"]:
                    raise ValueError("A completed checkpoint/artifact changed")
                continue
            state.update(current_task=task["id"], updated_at=now())
            entry.update(state="running", started_at=now())
            write_json(output / "status.json", state)
            print(f"[{state['completed_tasks']}/{state['total_tasks']}] {task['id']}", flush=True)
            log = output / "logs" / (task["id"].replace("/", "__") + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                print("Task attempt", now(), task["id"], flush=True)
                task_runner(output, plan, inputs, repo, task)
            artifact = verifier(output, plan, inputs, task)
            entry.update(state="complete", finished_at=now(), artifact=artifact)
            entry.pop("error", None)
            state["completed_tasks"] = sum(t["state"] == "complete" for t in state["tasks"].values())
            write_json(output / "status.json", state)
        except BaseException as exc:
            entry.update(state="failed", error=f"{type(exc).__name__}: {exc}", failed_at=now())
            state.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", updated_at=now())
            state["completed_tasks"] = sum(t["state"] == "complete" for t in state["tasks"].values())
            write_json(output / "status.json", state)
            error = output / "logs" / (task["id"].replace("/", "__") + ".error.log")
            error.parent.mkdir(parents=True, exist_ok=True)
            error.write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            gc.collect()
            if "torch" in sys.modules:
                torch = sys.modules["torch"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    final = []
    for task in plan["tasks"]:
        if task["stage"] == "refit":
            artifact = state["tasks"][task["id"]]["artifact"]
            final.append(dict(profile=task["profile"], group=task["group"], seed=task["seed"], path=task["path"],
                checkpoint_sha=artifact["hashes"]["best.pt"], **{k: artifact[k] for k in ("best_epoch", "parameter_count", "training_seconds")}))
    if len(final) != len(plan["profiles"]) * len(plan["groups"]) * len(plan["seeds"]):
        raise ValueError("Final model matrix is incomplete")
    write_json(output / "final.json", dict(status="complete", plan_sha=plan["plan_sha"], code_sha=plan["code_sha"],
        data_sha=plan["data_sha"], cohort=inputs["cohort"], models=final, completed_at=now(), all_seeds_reported=True))
    state.update(state="complete", current_task=None, finished_at=now())
    write_json(output / "status.json", state)
    print("All training complete:", len(final), "final models", flush=True)


def run(args):
    import torch
    from data import Repository, apply_shared_cohort
    output = args.output.resolve()
    with exclusive_lock(output):
        plan = read_json(output / "plan.json")
        verify_plan(plan)
        inputs = read_json(output / "inputs.json")
        if digest(inputs) != plan["inputs_sha"]:
            raise ValueError("Frozen preprocessing/split/cohort changed")
        repo = Repository(Path(plan["data_root"]), contract())
        if digest(repo.manifest) != plan["data_sha"] or apply_shared_cohort(repo, plan["profiles"]) != inputs["cohort"]:
            raise ValueError("Prepared data or common origins changed")
        if not torch.cuda.is_available():
            raise RuntimeError("Training requires CUDA; plan/init/tests don't train")
        execute_queue(output, plan, inputs, repo, args.resume)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("plan", "init"):
        q = sub.add_parser(name)
        q.add_argument("--profiles", nargs="+")
        q.add_argument("--feature-manifest", type=Path, action="append", metavar="JSON",
                       help="ordered custom input contract; repeat to add multiple custom profiles")
        q.add_argument("--output-policy", choices=["old128"], default="old128")
        q.add_argument("--seeds", nargs="+", type=int)
        q.add_argument("--epochs", type=int, default=50)
        q.add_argument("--patience", type=int, default=8)
        q.add_argument("--train-cap", type=int, default=0)
        q.add_argument("--workers", type=int, default=0)
    sub.choices["init"].add_argument("--data-root", type=Path, required=True)
    sub.choices["init"].add_argument("--output", type=Path, required=True)
    q = sub.add_parser("run")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--resume", action="store_true")
    q = sub.add_parser("status")
    q.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "plan":
        describe(make_plan(args.profiles, args.output_policy, args.seeds, args.epochs, args.patience,
                           args.train_cap, args.workers, args.feature_manifest))
    elif args.command == "init":
        initialize(args)
    elif args.command == "run":
        run(args)
    else:
        state = read_json(args.output / "status.json")
        print({k: state.get(k) for k in ("state", "completed_tasks", "total_tasks", "current_task", "updated_at")})
        for name, t in state["tasks"].items():
            if t["state"] == "failed":
                print("FAILED", name, t["error"])
        if state.get("current_task"):
            plan = read_json(args.output / "plan.json")
            task = next(t for t in plan["tasks"] if t["id"] == state["current_task"])
            history_path = args.output / task["path"] / "history.json"
            if history_path.exists():
                history = read_json(history_path)
                if history:
                    print("Latest completed epoch:", history[-1])


if __name__ == "__main__":
    main()

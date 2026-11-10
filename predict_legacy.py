"""Predict with a newly trained 10s-gap A/B/C/D model (27 outputs); no data repository required."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from forecast_windows import contiguous_segments

ROOT = Path(__file__).resolve().parent
GROUPS = dict(A='A_128_history_to_27', B='B_133_history_to_27',
              C='C_129_history_plus_4_conditions_to_27', D='D_129_history_plus_5_conditions_to_27')


def load_engine(group, directory, device='cpu'):
    directory = Path(directory).resolve()
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((directory/'feature_manifest.json').read_text(encoding='utf-8'))
    expected = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'manifest_sha256'},
                                        ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    if manifest.get('manifest_sha256') != expected or manifest.get('latency_gap_points') != 5:
        raise ValueError('Changed or incompatible source manifest')
    signature = config.get('contract', config)
    if (signature.get('forecast_gap_seconds') != 10 or signature.get('forecast_gap_points') != 5
            or signature.get('cadence_seconds') != 2):
        raise ValueError('Requires a newly trained 10s-gap model; historical30s/unknown-gap weights are not compatible')
    if config.get('manifest_sha256') != manifest.get('manifest_sha256'):
        raise ValueError('Training config and data manifest differ')
    here = ROOT/'experiments/latency_abcd_27'/GROUPS[group]
    sys.path.insert(0, str(here))
    file = 'latency_133_to_27_common.py' if group == 'B' else 'experiment_common.py'
    module_name = '_legacy_prediction_' + group
    spec = importlib.util.spec_from_file_location(module_name, here/file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    fields = manifest['feature_order']
    with np.load(directory/'scaler.npz', allow_pickle=False) as scaler:
        mean, scale = scaler['mean'].copy(), scaler['scale'].copy()
    if (mean.shape != (len(fields),) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('Invalid training scaler')
    if group == 'B':
        c = module.TargetContract(here/'selected_targets_27.json', SimpleNamespace(feature_names=fields))
        if config['input_feature_order'] != fields or config['output_feature_order'] != list(c.names):
            raise ValueError('Checkpoint feature order differs')
        model = module.SelectedOutputCrossformer(None, config['params'], c.input_indices)
        dynamic, conditions, targets = fields, [], list(c.names)
    else:
        c = module.ExperimentContract(here/'experiment_spec.json', directory)
        if config['contract'] != c.describe():
            raise ValueError('Checkpoint model role or target contract differs')
        model = module.ExperimentModel(c, None, config['params'], mean, scale)
        dynamic, conditions, targets = c.dynamic_features, c.condition_features, c.target_features
    state = torch.load(directory/'model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return SimpleNamespace(model=model, fields=fields, dynamic=dynamic, conditions=conditions,
                           targets=targets, mean=mean, scale=scale, device=torch.device(device), group=group)


def predict(group, directory, input_path, output, mode='latest', stride=5, device='cpu'):
    output = Path(output).resolve()
    directory, input_path = Path(directory).resolve(), Path(input_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == input_path or directory == output or directory in output.parents or output in directory.parents:
        raise ValueError('Output overlaps the model or input')
    if stride < 1 or mode not in ('latest', 'rolling'):
        raise ValueError('Invalid mode or stride')
    engine = load_engine(group, directory, device)
    frame = pd.read_csv(input_path) if input_path.suffix.lower() == '.csv' else pd.read_parquet(input_path)
    selected = list(dict.fromkeys(engine.dynamic + engine.conditions))
    if set(selected) - set(frame):
        raise ValueError('Missing selected input fields: '+str(sorted(set(selected)-set(frame))))
    if not np.isfinite(frame[selected].to_numpy(dtype='float32')).all():
        raise ValueError('Historical inputs must be finite after causal alignment')
    for name in ['target_observed_mask', 'observed_mask']:
        if name in frame and (frame[name].isna().any() or not frame[name].isin([0, 1, False, True]).all()):
            raise ValueError('Invalid observation mask')
    windows = []
    di, ci = [[engine.fields.index(f) for f in names] for names in (engine.dynamic, engine.conditions)]
    ti = [engine.fields.index(f) for f in engine.targets]
    for part, times in contiguous_segments(frame, 'time_utc'):
        n = len(part)
        starts = [n-150] if mode == 'latest' and n >= 150 else range(0, n-215+1, stride) if mode == 'rolling' else []
        for start in starts:
            end = start + 150
            mask = next((name for name in ['target_observed_mask', 'observed_mask'] if name in part), None)
            if mode == 'rolling' and mask and not part[mask].iloc[end+5:end+65].astype(bool).all():
                continue
            raw = np.zeros((150, len(engine.fields)), dtype='float32')
            for f in selected:
                raw[:, engine.fields.index(f)] = part[f].iloc[start:end].to_numpy(dtype='float32')
            scaled = (raw-engine.mean)/engine.scale
            identity = {f: part[f].iloc[end-1] for f in ['device_id', 'job_id', 'segment_id'] if f in part}
            windows.append((scaled, times.iloc[end-1], identity))
    if mode == 'latest':
        windows = windows[-1:]
    if not windows:
        raise ValueError('No eligible task-bounded input windows')
    records = []
    with torch.inference_mode():
        for values, origin, identity in windows:
            dynamic = torch.from_numpy(values[:, di]).unsqueeze(0).to(engine.device)
            if group == 'B':
                result = engine.model(dynamic)
            else:
                condition = torch.from_numpy(values[:, ci]).unsqueeze(0).to(engine.device)
                result = engine.model(dynamic, condition)
            result = result[0].float().cpu().numpy()*engine.scale[ti]+engine.mean[ti]
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite model output')
            result[:, 22:] = result[:, 22:] >= 0.5
            for step, row in enumerate(result, 1):
                record = dict(identity, forecast_origin_time=origin, horizon_step=step,
                              prediction_time=origin+pd.Timedelta(seconds=10+2*step))
                record.update(zip(engine.targets, row.tolist()))
                records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    if output.suffix.lower() == '.csv':
        result.to_csv(output, index=False)
    else:
        result.to_parquet(output, index=False, compression='zstd')
    return dict(output=str(output), contexts=len(windows), output_features=27, forecast_gap_seconds=10)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=list(GROUPS), required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['latest', 'rolling'], default='latest')
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    print(json.dumps(predict(a.group, a.model_dir, a.input, a.output, a.mode, a.stride, a.device), ensure_ascii=False, indent=2))

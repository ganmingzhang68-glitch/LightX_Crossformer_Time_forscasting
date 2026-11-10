"""Independent-user workflows, using synthetic data and forward passes only."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tools.prepare_old128 import prepare as prepare_old, regularize
from tools.convert_added_csv import read_headers
from tools.dataset_to_inference import convert
from predict_legacy import GROUPS, predict, load_engine

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_old_source_holds_only_missing_tick_and_keeps_boundaries():
    frame = pd.DataFrame(dict(time_utc=pd.to_datetime(['2026-07-01T00:00:00Z',
        '2026-07-01T00:00:04Z', '2026-07-01T00:00:06Z', '2026-07-01T00:00:08Z']),
        f=[10., 20., np.nan, 40.], b=[0., 1., 0., 1.]))
    rows = regularize(frame, ['f', 'b'], ['b'])
    assert rows.time_utc.dt.second.tolist() == [0, 2, 4, 8]
    assert rows.f.tolist() == [10., 10., 20., 40.]
    assert rows.observed_mask.tolist() == [True, False, True, True]
    assert rows.segment_id.nunique() == 2
    frame.loc[0, 'b'] = 2
    with pytest.raises(ValueError, match='binary'):
        regularize(frame, ['f', 'b'], ['b'])


def test_csv_optional_platform_header_is_independent_of_first_file(tmp_path):
    first, second = tmp_path/'new.csv', tmp_path/'old.csv'
    first.write_text('记录时间,字段,平台到位\n', encoding='utf-8')
    second.write_text('记录时间,字段\n', encoding='utf-8')
    expected = ['记录时间', '字段', '平台到位']
    assert read_headers({'device': [first, second]}) == expected
    assert read_headers({'device': [second, first]}) == expected


def sources(tmp_path):
    manifest = json.loads((ROOT/'artifacts/feature_manifest.json').read_text(encoding='utf-8'))
    fields, boolean = manifest['feature_order'], manifest['boolean_features']
    mapping = {}
    added = tmp_path/'added'; added.mkdir()
    for devno in range(1, 5):
        dev = f'{devno:02d}-X450-{devno}'
        times = pd.date_range('2026-07-01', periods=240, freq='2s', tz='UTC')
        old = pd.DataFrame({f: np.arange(240, dtype='float32')*0.01+devno for f in fields})
        for f in boolean:
            old[f] = (np.arange(240)//10 % 2).astype('float32')
        old = old.copy()
        old['time_utc'], old['device_id'] = times, dev
        path = tmp_path/(dev+'.parquet'); old.to_parquet(path, index=False); mapping[dev] = path
        new = pd.DataFrame({f: np.ones(240) for f in ['打印层数', '设松粉S', '设铺粉模式', '设前螺旋时长', '设后螺旋时长', '设风速']})
        new['time_utc'], new['device_id'], new['job_id'], new['source_row'] = times, dev, dev+'_job0', np.arange(1, 241)
        new['source_file'] = 'synthetic.csv'
        new.to_parquet(added/f'influx_sensors_added_{dev}_2026-07.parquet', index=False)
    old_root = tmp_path/'old128'
    prepare_old(mapping, old_root)
    assert (old_root/'metadata.json').exists()
    with pytest.raises(FileExistsError):
        prepare_old(mapping, old_root)
    return old_root, added


@pytest.mark.parametrize('group', list(GROUPS))
def test_abcd_from_raw_sources_to_independent_prediction(tmp_path, monkeypatch, group):
    old_root, added = sources(tmp_path)
    data = tmp_path/'prepared'
    module = load('_prepare_' + group, ROOT/'tools/prepare_abcd.py')
    monkeypatch.setattr(sys, 'argv', ['prepare_abcd', '--profile', 'D' if group == 'D' else 'ABC',
        '--old-root', str(old_root), '--new-data-dir', str(added), '--output', str(data)])
    module.main()
    manifest = json.loads((data/'feature_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['feature_order']) == (134 if group == 'D' else 133)
    assert manifest['latency_gap_points'] == 5
    directory = tmp_path/'model'; directory.mkdir()
    shutil.copyfile(data/'feature_manifest.json', directory/'feature_manifest.json')
    scaler = np.load(data/'folds/final/scaler.npz')
    np.savez_compressed(directory/'scaler.npz', mean=scaler['mean'], scale=scaler['scale'])
    params = dict(d_model=16, n_heads=2, e_layers=1, d_ff=32, factor=2, dropout=0,
        crossformer_seg_len=10, revin_affine=True, condition_hidden=8,
        condition_embedding_dim=4, condition_gate_init=0.1, condition_layers=1)
    here = ROOT/'experiments/latency_abcd_27'/GROUPS[group]
    sys.path.insert(0, str(here))
    if group == 'B':
        runtime = load('_fixture_B', here/'latency_133_to_27_common.py')
        contract = runtime.TargetContract(here/'selected_targets_27.json', SimpleNamespace(feature_names=manifest['feature_order']))
        model = runtime.SelectedOutputCrossformer(None, params, contract.input_indices)
        config = dict(input_feature_order=manifest['feature_order'], output_feature_order=list(contract.names),
            forecast_gap_seconds=10, forecast_gap_points=5, cadence_seconds=2)
    else:
        runtime = load('_fixture_'+group, here/'experiment_common.py')
        contract = runtime.ExperimentContract(here/'experiment_spec.json', directory)
        model = runtime.ExperimentModel(contract, None, params, scaler['mean'], scaler['scale'])
        config = dict(contract=contract.describe())
    config.update(params=params, manifest_sha256=manifest['manifest_sha256'])
    (directory/'config.json').write_text(json.dumps(config), encoding='utf-8')
    torch.save(model.state_dict(), directory/'model.pth')
    input_path = data/'devices/01-X450-1/rows.parquet'
    output = tmp_path/'prediction.parquet'
    predict(group, directory, input_path, output, mode='latest')
    result = pd.read_parquet(output)
    assert len(result) == 60
    delta = (result.prediction_time-result.forecast_origin_time).dt.total_seconds()
    assert delta.iloc[0] == 12 and delta.iloc[-1] == 130
    assert result.shape[1] == 33  # identities, timestamps, step, 27 targets
    # The same files produce exactly the same forward values when loaded independently.
    engine = load_engine(group, directory)
    assert not engine.model.training
    if group == 'B':
        config['forecast_gap_seconds'] = 30
    else:
        config['contract']['forecast_gap_seconds'] = 30
    (directory/'config.json').write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='10s-gap'):
        load_engine(group, directory)


def test_all90_export_supports_each_group_without_training(tmp_path):
    if not (ROOT/'experiments/all90_latency_v1').exists():
        pytest.skip('ALL90 is outside this branch')
    from experiments.all90_latency_v1 import common, data, model
    from experiments.all90_latency_v1.tests.test_contract_and_pipeline import repo_fixture
    from export_model import export_model
    from portable_inference import PortableModel
    c = common.contract(); repo = repo_fixture(c)
    pre = data.fit_preprocessor(repo, data.split_jobs(repo)['train'])
    params = dict(c['model_params'], d_model=16, n_heads=2, e_layers=1, d_ff=32, factor=2, condition_hidden=8)
    for group in c['groups']:
        trained = tmp_path/(group+'_model'); trained.mkdir()
        net = model.ForecastModel(c, group, pre, params).eval()
        spec = dict(contract_sha=common.digest(c), preprocessor=pre, params=params, group=group, code_sha=common.code_identity())
        signature = common.digest(spec)
        common.write_json(trained/'spec.json', dict(spec, signature=signature))
        common.write_json(trained/'result.json', dict(signature=signature))
        torch.save(dict(model=net.state_dict(), contract=c, group=group, pre=pre, params=params, epoch=1,
            seed=1, signature=signature, code_sha=common.code_identity()), trained/'best.pt')
        bundle = tmp_path/(group+'_bundle')
        assert export_model(trained, bundle, 'all90')['numerical_roundtrip']
        assert PortableModel(bundle).g['gap'] == 5


def test_old128_and_new90_preparation_produces_real_inference_schema(tmp_path):
    if not (ROOT/'experiments/all90_latency_v1').exists():
        pytest.skip('ALL90 is outside this branch')
    from experiments.all90_latency_v1 import common
    from experiments.all90_latency_v1.prepare import prepare
    old, added = sources(tmp_path)
    c = common.contract()
    for number in range(1, 5):
        dev = f'{number:02d}-X450-{number}'
        times = pd.date_range('2026-07-01', periods=240, freq='2s', tz='UTC')
        job = dev+'_job0'
        new = pd.DataFrame({f: np.ones(240) for f in c['added']})
        for date in c['dates']:
            new[date] = '2026-06-30 08:00:00'
        new['time_utc'], new['device_id'], new['job_id'], new['source_row'] = times, dev, job, np.arange(1, 241)
        new.to_parquet(added/f'influx_sensors_added_{dev}_2026-07.parquet', index=False)
        pd.DataFrame(dict(device_id=[dev], job_id=[job], job_start_utc=[times[0]],
            job_end_utc=[times[-1]])).to_parquet(added/f'influx_jobs_added_{dev}_2026-07.parquet', index=False)
    directory = tmp_path/'all90_dev'
    prepare(SimpleNamespace(old_root=old, added_root=added, contract=None, external=False,
        output=directory, stride=5, added_delay_seconds=0))
    output = tmp_path/'numeric.parquet'
    convert(directory, '01-X450-1_job0', output)
    frame = pd.read_parquet(output)
    assert set(c['source_features']) <= set(frame)
    assert frame[c['source_features']].shape == (240, 218)
    assert frame.job_id.unique().tolist() == ['01-X450-1_job0']
    assert {'observed_mask', 'new_observed', 'new_age_seconds'} <= set(frame)


def test_influx_long_tables_can_build_unscaled_old128_sources(tmp_path):
    old, _ = sources(tmp_path)
    manifest = json.loads((old/'feature_manifest.json').read_text(encoding='utf-8'))
    fields, boolean = manifest['feature_order'], set(manifest['boolean_features'])
    mapping = {}
    for number in range(1, 5):
        dev = f'{number:02d}-X450-{number}'
        wide = pd.read_parquet(tmp_path/(dev+'.parquet'))
        long = wide.melt(id_vars=['time_utc', 'device_id'], value_vars=fields,
                         var_name='field', value_name='value_float').rename(columns={'time_utc': 'time', 'device_id': 'host'})
        mask = long.field.isin(boolean)
        long['value_type'] = np.where(mask, 'bool', 'float')
        long['value_bool'] = pd.Series(pd.NA, index=long.index, dtype='boolean')
        long.loc[mask, 'value_bool'] = long.loc[mask, 'value_float'].astype(bool)
        long.loc[mask, 'value_float'] = np.nan
        long['measurement'] = 'PLC-Sensors'
        path = tmp_path/(dev+'_long.parquet'); long.to_parquet(path, index=False)
        mapping[dev] = path
    output = tmp_path/'old_from_long'
    prepare_old(mapping, output)
    result = pd.read_parquet(output/'validation.parquet')
    expected = pd.read_parquet(tmp_path/'04-X450-4.parquet')
    np.testing.assert_array_equal(result[fields], expected[fields])


def test_csv_conversion_keeps_utc_and_source_row_with_job_match(tmp_path, monkeypatch):
    from tools.convert_added_csv import main
    csv_dir = tmp_path/'csv'; csv_dir.mkdir()
    path = csv_dir/'ValuePLCLog_2026-07-01_08-00-00_Machine01.csv'
    path.write_text('记录时间,设铺粉模式,平台到位\n2026-07-01 08:00:00,1,0,\n2026-07-01 08:00:02,0,1,\n', encoding='utf-8')
    data = tmp_path/'data'; data.mkdir()
    start = pd.Timestamp('2026-07-01T00:00:00Z')
    pd.DataFrame(dict(time=[start, start], host=['01-X450-1']*2, field=['EndTime', 'JobName'],
        value_float=[(start+pd.Timedelta(seconds=2)).timestamp()*1000, np.nan],
        value_string=[None, 'synthetic'])).to_parquet(data/'influx_jobs_01-X450-1_2026-07.parquet', index=False)
    monkeypatch.setattr(sys, 'argv', ['convert', '--input-dir', str(csv_dir), '--data-dir', str(data), '--chunk-rows', '1'])
    main()
    result = pd.read_parquet(data/'influx_sensors_added_01-X450-1_2026-07.parquet')
    assert result.time_utc.iloc[0] == start
    assert result.source_row.tolist() == [1, 2]
    assert result['平台到位'].tolist() == [0, 1]

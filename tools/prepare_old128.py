"""Build unscaled old128 source splits from explicit per-device Parquets; never trains."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def regularize(frame, features, boolean):
    """Causal whole-snapshot hold for one absent tick; explicit invalid rows stay invalid."""
    frame = frame.copy()
    frame['time_utc'] = pd.to_datetime(frame['time_utc'], utc=True)
    if frame.time_utc.isna().any():
        raise ValueError('Unknown source timestamps')
    frame = frame.sort_values('time_utc', kind='stable')
    if frame.time_utc.duplicated().any():
        raise ValueError('Duplicate source timestamps; resolve their meaning before preparing data')
    if frame.time_utc.dt.floor('2s').ne(frame.time_utc).any():
        raise ValueError('Old128 snapshots must already lie on the UTC 2s grid')
    for field in features:
        frame[field] = pd.to_numeric(frame[field], errors='raise')
    for field in boolean:
        if not frame[field].dropna().isin([0, 1]).all():
            raise ValueError('Non-binary source field: ' + field)
    boundary = frame.time_utc.diff().dt.total_seconds().gt(4)
    for identity in ['segment_id', 'job_id']:
        if identity in frame:
            if frame[identity].isna().any():
                raise ValueError('Unknown source identity: ' + identity)
            boundary |= frame[identity].ne(frame[identity].shift())
    pieces = []
    segment = 0
    for _, part in frame.groupby(boundary.cumsum(), sort=False):
        source = part.set_index('time_utc')[features]
        grid = pd.date_range(source.index[0], source.index[-1], freq='2s')
        z = source.reindex(grid)
        present = pd.Series(grid.isin(source.index), index=grid)
        observed = present & np.isfinite(z).all(axis=1)
        # Do not turn an explicit invalid observation into a valid reading.
        held = z.ffill(limit=1)
        z.loc[~present] = held.loc[~present]
        valid = np.isfinite(z).all(axis=1)
        labels = (~valid | ~valid.shift(fill_value=False)).cumsum()
        for _, run in z.loc[valid].groupby(labels.loc[valid], sort=False):
            out = run.astype('float32').reset_index(names='time_utc')
            out['time_beijing'] = out.time_utc.dt.tz_convert('Asia/Shanghai')
            out['observed_mask'] = observed.loc[run.index].to_numpy(bool)
            out['segment_id'] = np.int32(segment)
            segment += 1
            pieces.append(out)
    if not pieces:
        raise ValueError('No finite old128 snapshots remain')
    return pd.concat(pieces, ignore_index=True)


def prepare(sources, output, manifest_path=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = Path(manifest_path or ROOT/'artifacts/feature_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fields, boolean = manifest['feature_order'], manifest['boolean_features']
    if len(fields) != 128 or len(set(fields)) != 128:
        raise ValueError('Exactly the original ordered128 contract is required')
    expected = {f'{n:02d}-X450-{n}' for n in range(1, 5)}
    sources = {dev: Path(path).resolve() for dev, path in sources.items()}
    if set(sources) != expected:
        raise ValueError('Supply exactly devices01/02/03/04 using --source DEVICE=PATH')
    for path in [manifest_path.resolve(), *sources.values()]:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('Output overlaps an input')
    # Fail before writing if a source is absent.
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    writers, schema, summaries = {}, None, {}
    try:
        for dev, path in sorted(sources.items()):
            names = set(pq.ParquetFile(path).schema_arrow.names)
            if set(fields) <= names:
                cols = ['time_utc' if 'time_utc' in names else 'time', *fields]
                cols += [f for f in ['device_id', 'host', 'segment_id', 'job_id'] if f in names]
                frame = pd.read_parquet(path, columns=cols).rename(columns={'time': 'time_utc'})
            else:
                import duckdb
                from prepare_influx import build_wide, sql_string
                with duckdb.connect() as connection:
                    if 'host' not in names:
                        raise ValueError('Influx source requires host identity')
                    hosts = connection.execute('SELECT DISTINCT host FROM read_parquet('+sql_string(str(path))+')').fetchall()
                    if hosts != [(dev,)]:
                        raise ValueError('Source device differs from --source: '+dev)
                    duplicates = connection.execute('SELECT count(*) FROM (SELECT time, field FROM read_parquet('+sql_string(str(path))+") WHERE measurement='PLC-Sensors' GROUP BY time, field HAVING count(*)>1)").fetchone()[0]
                    if duplicates:
                        raise ValueError('Duplicate Influx field timestamps; resolve them before pivoting')
                with tempfile.TemporaryDirectory(prefix='old128_') as temp:
                    wide = Path(temp)/'wide.parquet'
                    build_wide(path, wide, fields, boolean)
                    frame = pd.read_parquet(wide).rename(columns={'time': 'time_utc'})
            for identity in ['device_id', 'host']:
                if identity in frame and set(frame[identity]) != {dev}:
                    raise ValueError('Source device identity mismatch: '+dev)
            frame = regularize(frame, fields, boolean)
            frame['device_id'] = dev
            split = 'validation' if dev == '04-X450-4' else 'train'
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if schema is None:
                schema = table.schema
                for name in ['train', 'validation', 'test']:
                    writers[name] = pq.ParquetWriter(output/(name+'.parquet'), schema, compression='zstd')
            writers[split].write_table(table.cast(schema))
            summaries[dev] = dict(rows=len(frame), observed=int(frame.observed_mask.sum()),
                                  segments=int(frame.segment_id.nunique()), source=str(path),
                                  sha256=file_hash(path))
    finally:
        for writer in writers.values():
            writer.close()
    manifest['continuous_values_clipped'] = False
    (output/'feature_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'metadata.json').write_text(json.dumps(dict(status='complete', sources=summaries,
        policy='No scaler, clipping, interpolation or task selection. One causal hold only into an absent tick.',
        splits='storage only: 01/02/03 train;04 validation;test empty; experiments re-split whole tasks'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, help='DEVICE=PATH (repeat four times)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'artifacts/feature_manifest.json')
    args = parser.parse_args()
    pairs = [s.split('=', 1) for s in args.source]
    if any(len(p) != 2 for p in pairs) or len(dict(pairs)) != len(pairs):
        parser.error('Use unique DEVICE=PATH pairs')
    print(json.dumps(prepare(dict(pairs), args.output, args.manifest), ensure_ascii=False, indent=2))

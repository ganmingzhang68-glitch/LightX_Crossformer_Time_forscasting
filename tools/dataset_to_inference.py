"""Export one prepared ALL90/matrix task into its numeric inference input; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def convert(data_root, job_id, output):
    root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Output must be outside the dataset')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Dataset is incomplete')
    matches = [r for r in manifest['jobs'] if r['job_id'] == job_id]
    if len(matches) != 1:
        raise ValueError('Select exactly one job_id from manifest.json')
    record = matches[0]
    directory = (root/record['path']).resolve()
    if root not in directory.parents:
        raise ValueError('Task path escapes dataset')
    for name in ['values.npy', 'rows.parquet']:
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != record['hashes'][name]:
            raise ValueError('Changed dataset file: '+name)
    rows = pd.read_parquet(directory/'rows.parquet')
    values = np.load(directory/'values.npy', mmap_mode='r', allow_pickle=False)
    if values.shape != (len(rows), len(manifest['source_features'])):
        raise ValueError('Stored source order or dimensions differ')
    result = pd.concat([rows, pd.DataFrame(values, columns=manifest['source_features'])], axis=1)
    result['device_id'], result['job_id'] = record['device'], job_id
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == '.csv':
        result.to_csv(output, index=False)
    else:
        result.to_parquet(output, index=False, compression='zstd')
    return dict(output=str(output), rows=len(rows), features=values.shape[1], job_id=job_id)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(convert(a.data_root, a.job_id, a.output), ensure_ascii=False, indent=2))

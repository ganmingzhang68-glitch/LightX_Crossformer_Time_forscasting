"""Check independent-clone imports, plans and optional published-model checksums; never trains."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check(training=False, models=False):
    results = dict(python=sys.version.split()[0], root=str(ROOT), dependencies={}, plans={}, errors=[],
                   training_started=False, optimizer_steps=0, cuda_training_verified=False)
    modules = ['numpy', 'pandas', 'pyarrow', 'duckdb', 'torch'] + (['optuna'] if training else [])
    for name in modules:
        try:
            module = importlib.import_module(name)
            results['dependencies'][name] = getattr(module, '__version__', 'installed')
            if name == 'torch':
                results['cuda_available'] = module.cuda.is_available()
                module.set_num_threads(1)
        except ImportError as e:
            results['errors'].append(str(e))
    if results['errors']:
        return results
    for directory, script in [('all90_latency_v1', 'run.py'), ('input_window_matrix_v1', 'train_queue.py')]:
        path = ROOT/'experiments'/directory/script
        if path.is_file():
            run = subprocess.run([sys.executable, str(path), 'plan'], cwd=ROOT, capture_output=True,
                                 text=True, encoding='utf-8', errors='replace')
            results['plans'][directory] = run.returncode == 0
            if run.returncode:
                results['errors'].append(run.stderr[-2000:])
    run = subprocess.run([sys.executable, str(ROOT/'validate_bundle.py')], cwd=ROOT, capture_output=True,
                         text=True, encoding='utf-8', errors='replace',
                         env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    results['historical_epoch45_artifact_check'] = run.returncode == 0
    if run.returncode:
        results['errors'].append(run.stderr[-2000:])
    if models:
        directory = ROOT/'models/all90_20261110'
        if not directory.exists():
            results['published_models'] = 'Not stored in this branch; use the download instructions in USAGE.md'
        else:
            count = 0
            for line in (directory/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
                expected, path = line.split('  ', 1)
                actual = file_hash(directory/path)
                if expected != actual:
                    results['errors'].append('Model file checksum mismatch: '+path)
                count += 1
            results['published_model_files_checked'] = count
    results['cpu_installation_passed'] = not results['errors']
    results['cuda_training_environment_available'] = bool(training and results.get('cuda_available') and not results['errors'])
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', action='store_true', help='Also check HPO dependency and CUDA availability')
    p.add_argument('--models', action='store_true', help='Verify SHA-256 of all stored model files')
    a = p.parse_args()
    result = check(a.training, a.models)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['errors'] else 0)

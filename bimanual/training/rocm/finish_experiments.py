"""Run on the droplet: bound experiments, evaluate, stage a verified backup."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/root/bimanual-training')
JOBS = ('smolvla-mug-20k', 'compact-160m-2k', 'pi05-mug-smoke')


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def main():
    deadline = time.monotonic() + 50 * 60
    while True:
        states = {name: json.loads(run('docker', 'inspect', name, capture_output=True).stdout)[0]['State']
                  for name in JOBS}
        if not any(state['Running'] for state in states.values()):
            break
        if time.monotonic() >= deadline:
            for name, state in states.items():
                if state['Running']:
                    run('docker', 'stop', '--time', '30', name)
            break
        time.sleep(15)

    # An evaluation failure must not prevent artifact export or cost control.
    with (ROOT / 'logs/evaluation.log').open('w') as log:
        evaluation = subprocess.run([
            'docker', 'run', '--name', 'bimanual-heldout-eval', '--device=/dev/kfd',
            '--device=/dev/dri', '--group-add', 'video', '--shm-size=16g',
            '-v', f'{ROOT}:/workspace', 'bimanual-smolvla:rocm',
            'timeout', '--kill-after=30s', '480',
            'python', '-m', 'bimanual.training.rocm.evaluate',
        ], stdout=log, stderr=subprocess.STDOUT)
    statuses = {'evaluation_exit_code': evaluation.returncode,
                'pi05_scope': '200-step ROCm compatibility and fine-tuning smoke test',
                'custom_scope': 'action-only feasibility; auxiliary heads not trained',
                'droplet_id': 603294423,
                'billing_note': 'Power-off does not end billing; droplet deletion is required.'}
    for name in JOBS:
        state = json.loads(run('docker', 'inspect', name, capture_output=True).stdout)[0]['State']
        statuses[name] = {'status': state['Status'], 'exit_code': state['ExitCode']}
        with (ROOT / 'logs' / f'{name}.log').open('w') as log:
            run('docker', 'logs', name, stdout=log, stderr=subprocess.STDOUT)
    (ROOT / 'logs/experiment_status.json').write_text(json.dumps(statuses, indent=2)+'\n')

    export = ROOT / 'export'
    export.mkdir(exist_ok=False)
    # Only the latest full checkpoint is needed to resume; avoid copying all
    # intermediate checkpoints onto the laptop's limited disk.
    smol_root = ROOT / 'checkpoints/smolvla-mug-20k/checkpoints'
    checkpoints = sorted(p for p in smol_root.glob('[0-9]*') if p.is_dir())
    if checkpoints:
        shutil.copytree(checkpoints[-1], export / 'smolvla_latest')
    pi_checkpoints = sorted(p for p in (ROOT / 'checkpoints/pi05-mug-smoke/checkpoints').glob('[0-9]*')
                            if p.is_dir())
    if pi_checkpoints:
        shutil.copytree(pi_checkpoints[-1], export / 'pi05_latest')
    compact = ROOT / 'checkpoints/compact-160m-2k'
    if compact.exists():
        shutil.copytree(compact, export / 'compact_160m', ignore=shutil.ignore_patterns('*.tmp'))
    shutil.copytree(ROOT / 'logs', export / 'logs')
    shutil.copytree(ROOT / 'bimanual', export / 'bimanual', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'datasets/aloha_mug_train_stable_stats/meta', export / 'training_metadata')
    shutil.copy2(ROOT / 'datasets/aloha_mug_train_stable_stats/normalization_provenance.json', export)
    shutil.copy2(ROOT / 'train.py', export / 'smolvla_train.py')
    shutil.copytree(ROOT / 'build', export / 'build')
    manifest = {}
    for path in sorted(export.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                manifest[str(path.relative_to(export))] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (export / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (ROOT / 'EXPORT_READY').write_text('ready\n')
    print('EXPORT_READY', flush=True)


if __name__ == '__main__':
    main()

"""Publish private experiment artifacts using the local Hugging Face login.

Waits for the verified local backup; credentials never leave the local Hub SDK.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, get_token
from huggingface_hub.hf_api import RepoFile


def file_hashes(path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha256.update(chunk)
            git_blob.update(chunk)
    return size, sha256.hexdigest(), git_blob.hexdigest()


def upload_verified(api, repo_id, kind, root, prefix='', selected=None):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): p for p in root.rglob('*')
             if p.is_file() and not any(part in ('.cache', '__pycache__') for part in p.parts)
             and p.suffix != '.tmp'}
    if selected is not None:
        files = {name: path for name, path in files.items() if selected(name)}
    if not files:
        raise ValueError(f'No files to upload from {root}')
    api.create_repo(repo_id, repo_type=kind, private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type=kind).private:
        raise RuntimeError(f'Refusing upload to existing public repo: {repo_id}')
    print(f'Uploading {len(files)} files to {repo_id}/{prefix}', flush=True)
    commit = api.upload_folder(repo_id=repo_id, repo_type=kind, folder_path=root,
                               path_in_repo=prefix, allow_patterns=list(files),
                               commit_message='Preserve droplet experiment artifacts')
    remote = {p.path: p for p in api.list_repo_tree(
        repo_id, repo_type=kind, recursive=True, revision=commit.oid) if isinstance(p, RepoFile)}
    for name, path in files.items():
        remote_name = f'{prefix}/{name}' if prefix else name
        entry = remote.get(remote_name)
        size, sha256, blob = file_hashes(path)
        if entry is None or entry.size != size:
            raise RuntimeError(f'Hub file missing or size mismatch: {remote_name}')
        if entry.lfs:
            if entry.lfs.sha256 != sha256:
                raise RuntimeError(f'Hub SHA-256 mismatch: {remote_name}')
        elif entry.blob_id != blob:
            raise RuntimeError(f'Hub Git blob hash mismatch: {remote_name}')
    url = f'https://huggingface.co/{"datasets/" if kind == "dataset" else ""}{repo_id}'
    print(f'VERIFIED {url} @ {commit.oid}', flush=True)
    return {'url': url, 'revision': commit.oid, 'files_verified': len(files), 'path': prefix}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--artifacts', required=True)
    args = parser.parse_args()
    project, artifacts = Path(args.project).resolve(), Path(args.artifacts).resolve()
    status_file = artifacts.parent / 'huggingface_status.json'
    deadline = time.monotonic() + 2 * 60 * 60
    print('Waiting for local Hugging Face login with write access.', flush=True)
    while not get_token():
        if time.monotonic() >= deadline:
            raise RuntimeError('Hugging Face login was not configured before the two-hour deadline')
        time.sleep(15)
    api = HfApi()
    account = api.whoami()['name']
    result = {'account': account, 'private': True, 'uploads': [], 'complete': False}

    def record(value):
        result['uploads'].append(value)
        status_file.write_text(json.dumps(result, indent=2)+'\n')

    suffix = '603294423'
    for split in ('train', 'val'):
        record(upload_verified(api, f'{account}/bimanual-aloha-mug-{split}-{suffix}', 'dataset',
                               project / f'outputs/aloha_mug_{split}'))
    print('Datasets verified on Hub; waiting for final local checkpoint backup.', flush=True)
    while not (artifacts / 'BACKUP_VERIFIED').exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Final verified backup was not ready before deadline')
        time.sleep(15)
    smol = artifacts / 'smolvla_latest'
    record(upload_verified(api, f'{account}/bimanual-smolvla-mug-{suffix}', 'model',
                           smol / 'pretrained_model'))
    if (smol / 'training_state').exists():
        record(upload_verified(api, f'{account}/bimanual-smolvla-mug-{suffix}', 'model',
                               smol / 'training_state', prefix='training_state'))
    record(upload_verified(api, f'{account}/bimanual-compact-vla-160m-{suffix}', 'model',
                           artifacts / 'compact_160m'))
    pi = artifacts / 'pi05_latest'
    if pi.exists():
        record(upload_verified(api, f'{account}/bimanual-pi05-mug-smoke-{suffix}', 'model',
                               pi / 'pretrained_model'))
        if (pi / 'training_state').exists():
            record(upload_verified(api, f'{account}/bimanual-pi05-mug-smoke-{suffix}', 'model',
                                   pi / 'training_state', prefix='training_state'))
    record(upload_verified(api, f'{account}/bimanual-experiment-{suffix}', 'model', artifacts,
                           selected=lambda n: n.startswith(('logs/', 'bimanual/', 'build/', 'training_metadata/'))
                           or n in ('normalization_provenance.json', 'manifest.json', 'smolvla_train.py')))
    result['complete'] = True
    result['note'] = 'Private Hub uploads verified against local file hashes; droplet artifacts preserved.'
    status_file.write_text(json.dumps(result, indent=2)+'\n')
    (artifacts.parent / 'HUGGINGFACE_VERIFIED').write_text('All dataset/model/result uploads verified.\n')
    # User explicitly requested a completion notification on this Mac.
    subprocess.run(['osascript', '-e',
                    'display notification "Experiment artifacts are verified on Hugging Face. Droplet 603294423 can be deleted." with title "Bimanual experiments saved"'],
                   check=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

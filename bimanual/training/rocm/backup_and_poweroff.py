"""Run locally. Verify a result backup before requesting droplet power-off.

This does not delete the droplet or stop DigitalOcean billing.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True)
    parser.add_argument('--known-hosts', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--host', default='root@134.199.203.255')
    args = parser.parse_args()
    ssh = ['ssh', '-i', str(Path(args.key).resolve()), '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
           '-o', f'UserKnownHostsFile={Path(args.known_hosts).resolve()}']
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 75 * 60
    while time.monotonic() < deadline:
        probe = subprocess.run(ssh + [args.host, 'test -f /root/bimanual-training/EXPORT_READY'])
        if probe.returncode == 0:
            break
        time.sleep(20)
    else:
        raise RuntimeError('Export not ready before deadline; backup and early power-off were not performed')
    subprocess.run(['rsync', '-az', '--partial', '--timeout=120', '-e', shlex.join(ssh),
                    f'{args.host}:/root/bimanual-training/export/', str(destination) + '/'], check=True)
    manifest = json.loads((destination / 'manifest.json').read_text())
    for name, expected in manifest.items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination):
            raise ValueError('Invalid manifest path')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != expected:
            raise RuntimeError(f'Backup checksum mismatch: {name}')
    (destination / 'BACKUP_VERIFIED').write_text(f'{len(manifest)} files verified\n')
    print(f'BACKUP_VERIFIED: {destination}', flush=True)
    # Delay by one minute so SSH returns before shutdown. The remote 90-minute
    # failsafe stays in place in case this local process is interrupted.
    subprocess.run(ssh + [args.host, 'shutdown -h +1'], check=True)
    (destination / 'POWEROFF_SCHEDULED').write_text(
        'Power-off requested; delete droplet 603294423 to stop billing.\n')
    print('Power-off scheduled. DELETE DROPLET 603294423 TO STOP BILLING.', flush=True)


if __name__ == '__main__':
    main()

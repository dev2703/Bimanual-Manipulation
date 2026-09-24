import hashlib
import json
import subprocess
import sys

import pytest

from bimanual.training.rocm.backup_and_poweroff import main


@pytest.mark.parametrize('valid', [True, False])
def test_poweroff_requires_verified_backup(tmp_path, monkeypatch, valid):
    payload = b'checkpoint data'
    (tmp_path / 'model.pt').write_bytes(payload)
    digest = hashlib.sha256(payload if valid else b'corrupt').hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps({'model.pt': digest}))
    monkeypatch.setattr(sys, 'argv', ['backup', '--key', str(tmp_path / 'key'),
                                    '--known-hosts', str(tmp_path / 'hosts'),
                                    '--destination', str(tmp_path)])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', run)
    if valid:
        main()
        assert (tmp_path / 'BACKUP_VERIFIED').exists()
        assert calls[-1][-1] == 'shutdown -h +1'
    else:
        with pytest.raises(RuntimeError, match='checksum mismatch'):
            main()
        assert not (tmp_path / 'BACKUP_VERIFIED').exists()
        assert not any('shutdown -h +1' in call for call in calls)


def test_failed_transfer_never_requests_poweroff(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['backup', '--key', str(tmp_path / 'key'),
                                    '--known-hosts', str(tmp_path / 'hosts'),
                                    '--destination', str(tmp_path)])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == 'rsync':
            raise subprocess.CalledProcessError(23, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        main()
    assert not any('shutdown -h +1' in call for call in calls)

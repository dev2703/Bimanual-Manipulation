from types import SimpleNamespace

import pytest

from bimanual.training.rocm import publish_huggingface as publish


def test_upload_refuses_existing_public_repository(tmp_path):
    (tmp_path / 'model.pt').write_bytes(b'model')
    api = SimpleNamespace(create_repo=lambda *a, **k: None,
                          repo_info=lambda *a, **k: SimpleNamespace(private=False))
    with pytest.raises(RuntimeError, match='public repo'):
        publish.upload_verified(api, 'user/model', 'model', tmp_path)


@pytest.mark.parametrize('valid', [True, False])
def test_upload_verifies_large_file_content_hash(tmp_path, monkeypatch, valid):
    file = tmp_path / 'model.pt'
    file.write_bytes(b'model weights')
    size, sha256, _ = publish.file_hashes(file)
    class Entry:
        path = 'model.pt'
        lfs = SimpleNamespace(sha256=sha256 if valid else 'bad-hash')
    entry = Entry()
    entry.size = size
    monkeypatch.setattr(publish, 'RepoFile', Entry)
    api = SimpleNamespace(
        create_repo=lambda *a, **k: None,
        repo_info=lambda *a, **k: SimpleNamespace(private=True),
        upload_folder=lambda **k: SimpleNamespace(oid='revision'),
        list_repo_tree=lambda *a, **k: [entry],
    )
    if valid:
        result = publish.upload_verified(api, 'user/model', 'model', tmp_path)
        assert result['files_verified'] == 1
    else:
        with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
            publish.upload_verified(api, 'user/model', 'model', tmp_path)

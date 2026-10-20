"""The snapshot as data, not image: published by Databricks, read from Azure Blob Storage."""

import json
from pathlib import Path

import pytest

from heavden_api import snapshot
from heavden_api.config import Settings
from heavden_api.deps import build_state
from heavden_api.snapshot import (
    Snapshot,
    blob_reader,
    fetch_snapshot,
    file_listing,
    publish_snapshot,
)

URL = "https://heavdenjlytf.blob.core.windows.net/website/prod"


def reader(folder: Path):
    """What blob_reader gives on Container Apps, over a local folder standing in for the store."""
    read = []

    def read_file(path: str) -> bytes:
        read.append(path)
        return (folder / path).read_bytes()

    read_file.paths = read
    return read_file


def test_a_published_snapshot_is_downloaded_file_for_file(demo_snapshot, tmp_path):
    pointer = publish_snapshot(demo_snapshot, tmp_path / "store", "20261013-2300")
    read = reader(tmp_path / "store")
    folder = fetch_snapshot(read, tmp_path / "run")
    assert folder == tmp_path / "run" / "20261013-2300"
    assert file_listing(folder) == file_listing(demo_snapshot) == pointer["files"]
    assert read.paths[0] == "current.json"  # the pointer first, then the files it names
    assert Snapshot.load(folder).manifest == Snapshot.load(demo_snapshot).manifest


def test_a_new_snapshot_moves_the_pointer_and_keeps_the_old_one_for_rollback(
    demo_snapshot, tmp_path
):
    store = tmp_path / "store"
    publish_snapshot(demo_snapshot, store, "week-1")
    publish_snapshot(demo_snapshot, store, "week-2")
    assert json.loads((store / "current.json").read_text())["snapshot"] == "week-2"
    assert (store / "snapshots" / "week-1" / "manifest.json").exists()
    with pytest.raises(FileExistsError):  # a published snapshot is never changed
        publish_snapshot(demo_snapshot, store, "week-1")
    with pytest.raises(ValueError, match="snapshot name"):
        publish_snapshot(demo_snapshot, store, "../elsewhere")


def test_a_file_unlike_the_listing_is_refused(demo_snapshot, tmp_path):
    store = tmp_path / "store"
    publish_snapshot(demo_snapshot, store, "s1")
    (store / "snapshots" / "s1" / "manifest.json").write_text("{}")  # changed after publishing
    with pytest.raises(ValueError, match="manifest.json is not the file current.json lists"):
        fetch_snapshot(reader(store), tmp_path / "run")


@pytest.mark.parametrize(
    "pointer",
    [
        {"snapshot": "../elsewhere", "files": {}},
        {"snapshot": "s1", "files": {"../../etc/passwd": {"bytes": 1, "sha256": "x"}}},
        {"snapshot": "s1", "files": {"/etc/passwd": {"bytes": 1, "sha256": "x"}}},
    ],
)
def test_the_pointer_cannot_reach_outside_the_snapshot(tmp_path, pointer):
    (tmp_path / "current.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError):
        fetch_snapshot(reader(tmp_path), tmp_path / "run")


def test_blob_storage_is_read_with_the_identitys_token(monkeypatch):
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"file"

    def urlopen(request, timeout):
        sent.append((request.full_url, dict(request.header_items()), timeout))
        return Response()

    monkeypatch.setattr(snapshot.urllib.request, "urlopen", urlopen)
    tokens = iter(["t1", "t2"])
    read = blob_reader(URL + "/", lambda: next(tokens))
    assert read("snapshots/s1/gold/risk_scores.parquet") == b"file"
    read("current.json")  # a token per request: the identity's cache renews it
    (url, headers, timeout), second = sent
    assert url == URL + "/snapshots/s1/gold/risk_scores.parquet" and timeout == 60
    assert headers == {"Authorization": "Bearer t1", "X-ms-version": "2021-08-06"}
    assert second[1]["Authorization"] == "Bearer t2"


def test_the_api_starts_on_the_published_snapshot_without_databricks(demo_snapshot, tmp_path):
    publish_snapshot(demo_snapshot, tmp_path / "store", "20261013-2300")
    settings = Settings(
        snapshot_dir=tmp_path / "run",
        snapshot_url=URL,
        managed_identity="web-id",
        llm_provider="openai",
        openai_api_key=None,
    )
    state = build_state(settings, read_snapshot=reader(tmp_path / "store"))  # demo: no api
    assert state.snapshot.root == tmp_path / "run" / "20261013-2300"
    assert state.as_of == Snapshot.load(demo_snapshot).manifest.as_of_ts


def test_the_snapshot_url_is_only_read_as_the_managed_identity():
    with pytest.raises(ValueError, match="MANAGED_IDENTITY_CLIENT_ID"):
        Settings(snapshot_url=URL, managed_identity=None)

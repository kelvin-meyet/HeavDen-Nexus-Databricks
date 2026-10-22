"""The snapshot as data, not image: published by Databricks, read from Azure Blob Storage."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from heavden_api import snapshot
from heavden_api.config import Settings
from heavden_api.deps import build_state
from heavden_api.snapshot import (
    POINTER,
    Snapshot,
    blob_reader,
    fetch_snapshot,
    file_listing,
    point_to,
    publish_snapshot,
    staged_pointer,
)

URL = "https://heavdenjlytf.blob.core.windows.net/website/prod"
ROOT = Path(__file__).resolve().parents[1]


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
    assert read.paths[0] == POINTER == "current-v2.json"  # the pointer first, then its files
    assert Snapshot.load(folder).manifest == Snapshot.load(demo_snapshot).manifest


def test_a_new_snapshot_moves_the_pointer_and_keeps_the_old_one_for_rollback(
    demo_snapshot, tmp_path
):
    store = tmp_path / "store"
    publish_snapshot(demo_snapshot, store, "week-1")
    publish_snapshot(demo_snapshot, store, "week-2")
    assert json.loads((store / POINTER).read_text())["snapshot"] == "week-2"
    assert (store / "snapshots" / "week-1" / "manifest.json").exists()
    with pytest.raises(FileExistsError):  # a published snapshot is never changed
        publish_snapshot(demo_snapshot, store, "week-1")
    with pytest.raises(ValueError, match="snapshot name"):
        publish_snapshot(demo_snapshot, store, "../elsewhere")
    point_to(reader(store), writer(store), "week-1")  # the rollback
    assert fetch_snapshot(reader(store), tmp_path / "run").name == "week-1"


def writer(folder: Path):
    def write(path: str, data: bytes) -> None:
        (folder / path).write_bytes(data)

    return write


def test_a_release_serves_its_snapshot_only_once_pointed_at_it(demo_snapshot, tmp_path):
    # the release smoke-tests the new image on the snapshot before the website may load it
    store = tmp_path / "store"
    publish_snapshot(demo_snapshot, store, "served")
    publish_snapshot(demo_snapshot, store, "release", move_pointer=False)
    assert fetch_snapshot(reader(store), tmp_path / "now").name == "served"
    staged = fetch_snapshot(reader(store), tmp_path / "test", staged_pointer("release"))
    assert staged.name == "release" and Snapshot.load(staged)
    point_to(reader(store), writer(store), "release")
    assert fetch_snapshot(reader(store), tmp_path / "next").name == "release"


def test_an_image_reads_only_snapshots_in_its_own_format(demo_snapshot, tmp_path):
    # a newer release's export writes a pointer for its own format; an older image never sees it
    store = tmp_path / "store"
    publish_snapshot(demo_snapshot, store, "s1")
    old = json.loads((store / POINTER).read_text()) | {"schema_version": 1}
    (store / POINTER).write_text(json.dumps(old))
    with pytest.raises(ValueError, match="snapshot format 1, this code reads 2"):
        fetch_snapshot(reader(store), tmp_path / "run")


class FilesAPI:
    """Databricks' Files API as website_snapshot.py uses it, over a folder standing in for the
    volume (/Volumes/heavden_prod/ml/website)."""

    def __init__(self, folder: Path):
        self.folder, self.files = folder, self

    def _local(self, path: str) -> Path:
        prefix = "/Volumes/heavden_prod/ml/website/"
        assert path.startswith(prefix), path
        return self.folder / path.removeprefix(prefix)

    def download(self, path: str):
        return SimpleNamespace(contents=io.BytesIO(self._local(path).read_bytes()))

    def upload(self, path: str, contents, overwrite=None):
        assert overwrite  # moving the pointer replaces it
        self._local(path).write_bytes(contents.read())


def test_the_release_script_fetches_points_and_names_the_served_snapshot(demo_snapshot, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "website_snapshot", ROOT / "backend" / "scripts" / "website_snapshot.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    volume, api = "/Volumes/heavden_prod/ml/website", FilesAPI(tmp_path / "volume")
    publish_snapshot(demo_snapshot, tmp_path / "volume", "served")
    publish_snapshot(demo_snapshot, tmp_path / "volume", "release", move_pointer=False)
    staged = script.main(["fetch", volume, str(tmp_path / "test"), "--name", "release"], api)
    assert Path(staged).name == "release" and Snapshot.load(Path(staged))
    assert script.main(["current", volume], api) == "served"
    assert "now names release" in script.main(["point", volume, "release"], api)
    assert script.main(["current", volume], api) == "release"
    assert Path(script.main(["fetch", volume, str(tmp_path / "web")], api)).name == "release"


def test_a_file_unlike_the_listing_is_refused(demo_snapshot, tmp_path):
    store = tmp_path / "store"
    publish_snapshot(demo_snapshot, store, "s1")
    (store / "snapshots" / "s1" / "manifest.json").write_text("{}")  # changed after publishing
    with pytest.raises(ValueError, match=f"manifest.json is not the file {POINTER} lists"):
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
    (tmp_path / POINTER).write_text(json.dumps(pointer | {"schema_version": 2}))
    with pytest.raises(ValueError, match="not a snapshot|outside the snapshot"):
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

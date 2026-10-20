"""The website's published snapshots on their volume, as the Databricks identity in the environment
(in CI: prod's deploy identity, which may read and write the volume).

    python backend/scripts/website_snapshot.py fetch <volume> <into> [--name <snapshot>]
    python backend/scripts/website_snapshot.py point <volume> <snapshot>
    python backend/scripts/website_snapshot.py current <volume>

fetch:   download a snapshot (by default the one the website serves) through the checks the website
         makes at start-up (`heavden_api.snapshot.fetch_snapshot`); prints its folder. The release
         smoke-tests the new image on the snapshot it published but didn't point to yet.
point:   make <snapshot> the one the website serves from its next start (`point_to`); also the
         rollback, to an earlier snapshot's name.
current: the snapshot the website serves.

<volume> is e.g. /Volumes/heavden_prod/ml/website.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from heavden_api.snapshot import POINTER, fetch_snapshot, point_to, read_pointer, staged_pointer


def volume_files(api, volume: str):
    """read(path) and write(path, data) for files under `volume`, with the Files API."""
    root = volume.rstrip("/")

    def read(path: str) -> bytes:
        return api.files.download(f"{root}/{path}").contents.read()

    def write(path: str, data: bytes) -> None:
        api.files.upload(f"{root}/{path}", io.BytesIO(data), overwrite=True)

    return read, write


def main(argv: list[str] | None = None, api=None) -> str:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("volume")
    fetch.add_argument("into", type=Path)
    fetch.add_argument("--name", help="a published snapshot, served or not")
    point = commands.add_parser("point")
    point.add_argument("volume")
    point.add_argument("name")
    current = commands.add_parser("current")
    current.add_argument("volume")
    args = parser.parse_args(argv)

    if api is None:
        from databricks.sdk import WorkspaceClient

        api = WorkspaceClient()
    read, write = volume_files(api, args.volume)
    if args.command == "fetch":
        pointer = staged_pointer(args.name) if args.name else POINTER
        return str(fetch_snapshot(read, args.into, pointer))
    if args.command == "point":
        pointed = point_to(read, write, args.name)
        return f"{POINTER} now names {pointed['snapshot']}: served from the website's next start"
    return read_pointer(read)["snapshot"]


if __name__ == "__main__":
    print(main())

"""The API's image (backend/Dockerfile): public, so what goes into it is checked here."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "backend" / "Dockerfile"
IGNORE = ROOT / "backend" / "Dockerfile.dockerignore"


def rules() -> list[str]:
    lines = IGNORE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_only_listed_paths_reach_the_build():
    # an allowlist: everything is excluded, then the API's own files are let back in. Widening
    # it (e.g. `!data/`) would put .env-like or personal files one COPY away from a public image.
    assert rules()[0] == "*"
    assert [r for r in rules() if r.startswith("!")] == [
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!generator/",
        "!databricks/src/heavden/",
        "!backend/heavden_api/",
        "!docs/corpus/",
    ]


def test_the_image_holds_code_not_data():
    # the snapshot is downloaded at start-up (SNAPSHOT_URL): a new one needs no new image, and
    # the image builds from git alone
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert not [line for line in text.splitlines() if line.startswith("COPY") and "data/" in line]
    assert not [rule for rule in rules() if rule.startswith("!data/")]


def test_base_images_are_pinned_by_digest():
    text = DOCKERFILE.read_text(encoding="utf-8")
    images = re.findall(r"^ARG PYTHON=(\S+)", text, re.M) + [
        image for image in re.findall(r"^FROM (\S+)", text, re.M) if not image.startswith("$")
    ]
    assert len(images) == 2
    assert all(re.search(r":[\w.-]+@sha256:[0-9a-f]{64}$", image) for image in images), images


def test_the_image_installs_only_the_apis_libraries():
    syncs = re.findall(r"uv sync [^\n]+", DOCKERFILE.read_text(encoding="utf-8"))
    assert len(syncs) == 2
    for sync in syncs:
        assert "--locked --no-dev --group backend --group rag --no-editable" in sync.replace(
            " --no-install-project", ""
        )
        assert not re.search(r"--group (spark|agent|notebooks|azure)|--all-groups", sync)

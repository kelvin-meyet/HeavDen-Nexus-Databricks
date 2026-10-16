"""Keep the model card in step with the model in production.

`docs/corpus/model-card.md` (one of the documents the assistant answers from) names the version
it describes in its front matter (`model_version`). Two workflows use this script:

    model_card.py check <champion version>
        release.yml, before a prod deploy: fails while the card describes another version than
        prod's @champion, so a release never ships a stale card.
    model_card.py remind <model_ops result file> <promoted|rolled back>
        promote.yml / rollback.yml, after a prod alias move: opens a GitHub issue to reissue the
        card (unless the card already describes the version, or the same issue is open).

Standard library only (and the `gh` CLI for the issue): runs on a bare GitHub runner.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

CARD = Path(__file__).resolve().parents[2] / "docs" / "corpus" / "model-card.md"


def front_matter(path: Path = CARD) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if match is None:
        raise SystemExit(f"{path}: no front matter")
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields


def card_version(path: Path = CARD) -> str:
    meta = front_matter(path)
    if meta.get("status") != "current" or not meta.get("model_version"):
        raise SystemExit(f"{path}: needs status: current and a model_version")
    return meta["model_version"]


def check(champion: str, path: Path = CARD) -> str:
    """The line to print when the card describes `champion`; raises SystemExit otherwise."""
    card = card_version(path)
    if not champion:
        return f"No @champion yet; the model card describes version {card}."
    if card != champion:
        raise SystemExit(
            f"The model card ({path.relative_to(CARD.parents[2])}) describes version {card}, but "
            f"prod's @champion is version {champion}. Reissue the card for version {champion} "
            "(keep the old one as superseded), update the README's results, then release."
        )
    return f"The model card describes version {card}, prod's @champion."


def issue(result: dict, action: str) -> tuple[str, str]:
    """Title and body of the reminder for a version that became prod's @champion."""
    version, before = str(result["version"]), str(result.get("from_version") or "none")
    title = f"Model card: describe version {version} ({action} in prod)"
    gates = result.get("gates") or {}
    lines = [
        f"Version {version} became prod's `@champion` ({action}; it replaces version {before}, "
        f"approved by {result.get('actor') or 'unknown'}). The model card still describes "
        f"version {card_version()}, and the assistant answers from it.",
        "",
        "To do, in one pull request:",
        f"1. Copy `docs/corpus/model-card.md` to a superseded version (status `superseded`, "
        f"`superseded_by`), as `model-card-v{card_version()}.md`.",
        f'2. Reissue `model-card.md`: next card version, `model_version: "{version}"`, '
        "effective from the first hour the version scores; every figure from the version's "
        "MLflow run (training, validation and test blocks, test results against the previous "
        "champion and NEWS2), nothing estimated.",
        "3. Update the README's production results.",
        "4. Release: `release.yml` refuses to deploy while the card describes another version.",
    ]
    if gates:
        lines += ["", "Gate results on the test block (from the promotion):", "", "```json"]
        lines += [json.dumps(gates, indent=2, sort_keys=True), "```"]
    return title, "\n".join(lines)


def remind(result: dict, action: str, run=subprocess.run) -> str:
    version = str(result["version"])
    if card_version() == version:
        return f"The model card already describes version {version}."
    title, body = issue(result, action)
    found = run(
        [
            "gh",
            "issue",
            "list",
            "--state",
            "open",
            "--search",
            f'in:title "{title}"',
            "--json",
            "title",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if any(i["title"] == title for i in json.loads(found or "[]")):
        return f"Already open: {title}"
    run(["gh", "issue", "create", "--title", title, "--body", body], check=True)
    return f"Opened: {title}"


def main(argv: list[str]) -> None:
    if argv[:1] == ["check"] and len(argv) == 2:
        print(check(argv[1]))
    elif argv[:1] == ["remind"] and len(argv) == 3:
        print(remind(json.loads(Path(argv[1]).read_text(encoding="utf-8")), argv[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

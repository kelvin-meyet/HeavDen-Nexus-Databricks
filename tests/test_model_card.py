"""The model card follows the model in production (.github/scripts/model_card.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

from heavden.agent.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_card", ROOT / ".github/scripts/model_card.py")
model_card = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_card)

PROMOTED = {"version": "4", "from_version": "3", "actor": "kelvin", "gates": {"auroc": 0.8}}


def test_one_current_card_names_the_version_it_describes():
    cards = [d for d in load_corpus(ROOT / "docs" / "corpus") if d.doc_id == "HD-ML-001"]
    assert sorted(d.status for d in cards) == ["current", "superseded"]
    assert model_card.card_version() == "3"
    old = model_card.front_matter(ROOT / "docs/corpus/model-card-v1.md")
    assert old["model_version"] == "1" and old["superseded_by"].startswith("HD-ML-001 v2.0")


def test_a_release_is_refused_while_the_card_describes_another_version():
    assert "describes version 3" in model_card.check("3")
    assert "No @champion yet" in model_card.check("")
    with pytest.raises(SystemExit, match="describes version 3, but prod's @champion is version 4"):
        model_card.check("4")


def test_the_reminder_lists_the_work_and_the_gates():
    title, body = model_card.issue(PROMOTED, "promoted")
    assert title == "Model card: describe version 4 (promoted in prod)"
    assert 'model_version: "4"' in body and "model-card-v3.md" in body
    assert '"auroc": 0.8' in body and "release.yml" in body


class Gh:
    def __init__(self, open_titles: list[str]):
        self.open_titles, self.calls = open_titles, []

    def __call__(self, args, **kwargs):
        self.calls.append(args[:3])
        out = json.dumps([{"title": t} for t in self.open_titles])

        class Done:
            stdout = out

        return Done()


def test_a_promotion_opens_one_reminder():
    gh = Gh([])
    assert model_card.remind(PROMOTED, "promoted", run=gh).startswith("Opened")
    assert gh.calls == [["gh", "issue", "list"], ["gh", "issue", "create"]]
    again = Gh(["Model card: describe version 4 (promoted in prod)"])
    assert model_card.remind(PROMOTED, "promoted", run=again).startswith("Already open")
    assert again.calls == [["gh", "issue", "list"]]


def test_no_reminder_when_the_card_already_describes_the_version():
    gh = Gh([])
    assert "already describes" in model_card.remind({**PROMOTED, "version": 3}, "promoted", run=gh)
    assert gh.calls == []

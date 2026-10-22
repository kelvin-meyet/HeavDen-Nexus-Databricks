"""Which changes deploy the Databricks assistant (.github/scripts/assistant_changed.py)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/assistant_changed.py"
spec = importlib.util.spec_from_file_location("assistant_changed", SCRIPT)
assistant_changed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assistant_changed)


@pytest.mark.parametrize(
    "path",
    [
        "backend/heavden_api/chat/agent.py",
        "backend/scripts/redteam_chat.py",
        "docs/corpus/model-card.md",
        "databricks/src/heavden/agent/genie_space.yaml",
        "databricks/resources/assistant_deploy.job.yml",
        "databricks/src/jobs/publish_assistant.py",
        "databricks/databricks.yml",
        "uv.lock",
    ],
)
def test_the_assistant_s_own_files_deploy_it(path):
    assert assistant_changed.assistant_files([path]) == [path]


@pytest.mark.parametrize(
    "path",
    [
        "web/app/page.tsx",
        "backend/heavden_api/app.py",
        "databricks/src/heavden/ml/train.py",
        "docs/how-code-reaches-databricks.md",
        ".github/workflows/release.yml",
    ],
)
def test_other_changes_leave_it_alone(path):
    assert assistant_changed.assistant_files([path]) == []


def test_the_exit_code_answers_the_workflow():
    def run(names: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT)], input=names, capture_output=True, text=True
        )

    changed = run("web/app/page.tsx\ndocs/corpus/model-card.md\n")
    assert changed.returncode == 0 and changed.stdout.split() == ["docs/corpus/model-card.md"]
    unchanged = run("web/app/page.tsx\n")
    assert unchanged.returncode == 1 and unchanged.stdout == ""
    assert run("").returncode == 1


def test_staging_and_prod_deploy_it_on_the_same_changes():
    for workflow, job in [("ci.yml", "assistant-staging"), ("release.yml", "deploy-prod")]:
        text = (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
        steps = yaml.safe_load(text)["jobs"][job]["steps"]
        changed = next(s for s in steps if s.get("id") == "changed")
        assert "assistant_changed.py" in changed["run"]

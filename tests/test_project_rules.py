"""Rules this repository holds itself to, checked on every run so they cannot quietly lapse.

Each one is a lesson that cost time when it was only written down:
- an unpinned runner image changed overnight (22 Oct 2026) and broke the Spark tests;
- an action pinned to a tag, or a tool CI installs at "latest", can change the same way;
- a deploy path nobody listed ships code nobody approved;
- a public repository publishes a committed key at once;
- a skipped test makes a red suite look green.

Not checked yet: the build backend (`hatchling`) in pyproject.toml is not pinned. Pinning it
changes pyproject.toml, which also redeploys the Databricks assistant to staging
(.github/scripts/assistant_changed.py), so it waits for a change that does that anyway; its check
comes with it.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
PINNED_ACTION = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")  # owner/repo@<full commit hash>
SKIP_MARKERS = ("mark." + "skip", "mark." + "xfail", "pytest." + "skip(")  # split: not a match here


def test_workflows_exist() -> None:
    assert WORKFLOWS, "no CI workflow: nothing checks a change before it merges"


def test_runner_images_are_pinned() -> None:
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        runners = re.findall(r"runs-on:\s*(\S+)", text)
        # a job that only calls another workflow (prod-run, staging-run) has no runner of its own
        calls_a_workflow = re.search(r"^    uses:\s*\S+", text, re.MULTILINE)
        assert runners or calls_a_workflow, f"{workflow.name} names no runner"
        for runner in runners:
            assert "latest" not in runner, (
                f"{workflow.name}: `{runner}` moves without warning; name an exact image"
            )


def test_actions_are_pinned_to_a_commit() -> None:
    for workflow in WORKFLOWS:
        for action in re.findall(r"uses:\s*(\S+)", workflow.read_text(encoding="utf-8")):
            if not action.startswith("./"):
                assert PINNED_ACTION.match(action), (
                    f"{workflow.name}: pin `{action}` to a full commit hash"
                )


def test_uv_is_pinned_wherever_ci_installs_it() -> None:
    for workflow in WORKFLOWS:
        steps = workflow.read_text(encoding="utf-8").split("\n      - ")  # one piece per step
        for step in steps:
            if "astral-sh/setup-uv@" in step:
                assert re.search(r'^\s+version:\s*"\d+\.\d+\.\d+"', step, re.MULTILINE), (
                    f"{workflow.name}: setup-uv installs the latest uv unless given `version:`"
                )


def test_dependencies_install_only_from_the_lock_file() -> None:
    assert (ROOT / "uv.lock").is_file(), "commit uv.lock: every machine installs the same versions"
    syncs = [
        line
        for workflow in WORKFLOWS
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if re.search(r"\buv sync\b", line)
    ]
    assert syncs, "no workflow installs the project with `uv sync`"
    for line in syncs:
        assert "--locked" in line or "--frozen" in line, (
            f"`{line.strip()}` may install versions other than uv.lock's: add --locked"
        )


def test_every_workflow_is_listed_as_a_deploy_path_or_not() -> None:
    paths = ROOT / "docs" / "deploy-paths.md"
    assert paths.is_file(), "write docs/deploy-paths.md: every way code reaches an environment"
    listed = paths.read_text(encoding="utf-8")
    for workflow in WORKFLOWS:
        assert f"`{workflow.name}`" in listed, (
            f"say in docs/deploy-paths.md whether {workflow.name} deploys anything"
        )


def test_secrets_are_scanned_before_commit_and_in_ci() -> None:
    config = ROOT / ".pre-commit-config.yaml"
    assert config.is_file(), "add .pre-commit-config.yaml with gitleaks"
    assert "gitleaks" in config.read_text(encoding="utf-8"), "pre-commit does not run gitleaks"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"gitleaks.* git ", ci), "ci.yml does not scan the history for secrets"
    assert "fetch-depth: 0" in ci, "the secret scan needs the whole history (fetch-depth: 0)"


def test_no_test_file_skips_its_own_tests() -> None:
    """Tests that cannot run here (Delta needs Linux) are marked in conftest.py, in one place."""
    for test_file in (ROOT / "tests").glob("test_*.py"):
        text = test_file.read_text(encoding="utf-8")
        for marker in SKIP_MARKERS:
            assert marker not in text, (
                f"{test_file.name} skips a test ({marker}): use a marker from conftest.py, "
                "or fix the test or the code"
            )

"""What CI runs, read from .github/workflows/ci.yml."""

from pathlib import Path

import yaml

from heavden_api.chat.models import load_models

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
BUNDLE = ROOT / "databricks" / "databricks.yml"


def quality_job() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]["assistant-quality"]


def test_the_quality_gate_tests_every_model_the_databricks_assistant_may_use():
    # one job per model (a sign-in lasts about an hour; three models in one job took ~53 min)
    job = quality_job()
    assert job["strategy"]["matrix"]["model"] == list(load_models("databricks"))
    assert job["strategy"]["fail-fast"] is False


def test_only_the_model_staging_serves_can_fail_the_run():
    job = quality_job()
    served = [e["model"] for e in job["strategy"]["matrix"]["include"] if e.get("served")]
    bundle = yaml.safe_load(BUNDLE.read_text(encoding="utf-8"))
    staging = (bundle["targets"]["staging"] or {}).get("variables", {})
    default = bundle["variables"]["assistant_llm_endpoint"]["default"]
    assert served == [staging.get("assistant_llm_endpoint", default)]
    test = next(s for s in job["steps"] if s.get("name", "").startswith("Quality test"))
    assert "${{ matrix.served && '--gate' || '' }}" in test["run"]

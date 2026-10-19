"""What CI runs, read from .github/workflows/ci.yml."""

from pathlib import Path

import yaml

from heavden_api.chat.models import load_models

CI = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def test_the_quality_gate_tests_every_model_the_databricks_assistant_may_use():
    # one job per model (a sign-in lasts about an hour; three models in one job took ~53 min)
    job = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]["assistant-quality"]
    assert job["strategy"]["matrix"]["model"] == list(load_models("databricks"))
    assert job["strategy"]["fail-fast"] is False

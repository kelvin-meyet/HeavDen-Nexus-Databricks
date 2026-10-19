"""Did the Databricks assistant change? Reads changed file names (`git diff --name-only`) on stdin,
prints the ones the assistant is built from and exits 0 if there are any, 1 if there are none.

ci.yml asks it on every merge (staging) and release.yml since the previous release (prod), so both
environments deploy the assistant on the same changes.
"""

import re
import sys

# its code, documents, Genie space, jobs, settings and dependencies
ASSISTANT = re.compile(
    r"^(backend/heavden_api/chat/"
    r"|backend/scripts/(assistant_quality|redteam_chat)\.py"
    r"|docs/corpus/"
    r"|databricks/src/heavden/agent/"
    r"|databricks/resources/assistant"
    r"|databricks/src/jobs/(deploy|publish)_assistant\.py"
    r"|databricks/databricks\.yml"
    r"|pyproject\.toml|uv\.lock)"
)


def assistant_files(paths: list[str]) -> list[str]:
    return [path for path in paths if ASSISTANT.match(path)]


if __name__ == "__main__":
    files = assistant_files(sys.stdin.read().splitlines())
    for path in files:
        print(path)
    sys.exit(0 if files else 1)

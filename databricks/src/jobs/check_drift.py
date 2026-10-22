"""Job entry point: compare the latest data, predictions and outcomes with the champion's
training data, and log every signal to the monitoring schema.

Run by the bundle job `drift_check` (`databricks/resources/drift_check.job.yml`) on serverless,
with the project wheel installed; the logic is `heavden.monitoring.drift_check`, tested locally.
"""

import sys

from heavden.monitoring.drift_check import main

main(sys.argv[1:])

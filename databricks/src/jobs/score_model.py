"""Job entry point: score every patient-hour with the registered model and build the Gold
risk, alert, escalation and KPI tables.

Run by the bundle job `score` (`databricks/resources/score.job.yml`) on serverless, with the
project wheel installed; the logic is `heavden.ml.batch_scoring`, tested locally.
"""

import sys

from heavden.ml.batch_scoring import main

main(sys.argv[1:])

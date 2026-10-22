"""Job entry point: promote a challenger, roll the champion back, or record decisions, each with
a row in the audit table.

Run by the bundle job `model_ops` (`databricks/resources/model_ops.job.yml`) on serverless, with
the project wheel installed; the logic is `heavden.ml.model_ops`, tested locally.
"""

import sys

from heavden.ml.model_ops import main

main(sys.argv[1:])

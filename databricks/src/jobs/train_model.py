"""Job entry point: train, evaluate and register the deterioration-risk model.

Run by the bundle job `train` (`databricks/resources/train.job.yml`) on serverless, with the
project wheel installed; the logic is `heavden.ml.training`, tested locally.
"""

import sys

from heavden.ml.training import main

main(sys.argv[1:])

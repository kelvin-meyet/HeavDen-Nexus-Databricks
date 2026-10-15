"""Job entry point: publish what the assistant reads on Databricks (the documents as
`ml.corpus_chunks`, the function `ml.explain_patient_risk`).

Run by the bundle job `assistant_publish` (`databricks/resources/assistant.job.yml`) on serverless,
with the project wheel installed; the logic is `heavden.agent.publish`, tested locally.
"""

import sys

from heavden.agent.publish import main

main(sys.argv[1:])

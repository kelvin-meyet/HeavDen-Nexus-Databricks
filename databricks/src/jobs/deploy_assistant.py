"""Job entry point: log, register and deploy the Databricks assistant, then ask it one question.

Run by the bundle job `assistant_deploy` (`databricks/resources/assistant_deploy.job.yml`) on
serverless, with the project wheel installed; the logic is `heavden_api.chat.assistant_deploy`,
tested locally.
"""

import sys

from heavden_api.chat.assistant_deploy import main

main(sys.argv[1:])

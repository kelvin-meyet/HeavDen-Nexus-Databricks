"""Job entry point: export the website's snapshot from Gold, check it, and publish it to the
website's volume.

Run by the bundle job `website_snapshot` (`databricks/resources/website_snapshot.job.yml`) on
serverless, with the project wheel installed; the logic is `heavden_api.export`, tested locally.
"""

import sys

from heavden_api.export import main

main(sys.argv[1:])

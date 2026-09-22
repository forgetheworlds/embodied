"""Entry point for ``python -m embodied``.

The exit code comes straight from the dispatcher: 0 for a valid complete result,
1 for a command error, 2 when a prerequisite is missing, 3 while a mandatory
adjudication is pending.
"""

import sys

from embodied.cli import main

if __name__ == "__main__":
    sys.exit(main())

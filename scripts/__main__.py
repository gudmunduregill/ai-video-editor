"""Entry point for running scripts as a module: python -m scripts"""

import sys

from scripts.cli import main


if __name__ == "__main__":
    sys.exit(main())

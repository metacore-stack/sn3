"""Entry point for ``python -m fineweb_loader``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

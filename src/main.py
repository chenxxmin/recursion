import os
import sys
from core import run_experiment


def main(config_path=None):
    run_experiment(config_path)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)

import importlib
import sys
from pathlib import Path


def set_system_path():
    package = importlib.import_module("nanotron")
    # NOTE:  Path(package.__file__).parent = .../nanotron/src/nanotron
    # we want .../nanotron
    package_path = Path(package.__file__).parent.parent.parent
    sys.path.append(str(package_path))

import sys
from pathlib import Path

from datasets import Dataset


def set_sys_path():
    current_script_dir = Path(__file__).resolve().parent
    # Calculate the root directory based on the current directory structure
    project_root = current_script_dir.parent

    # Add the project root to sys.path
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


def create_dummy_dataset(num_items: int):
    data = {"text": list(range(num_items))}
    return Dataset.from_dict(data)

import shutil
import uuid
from functools import lru_cache
from pathlib import Path


class TestContext:
    def __init__(self):
        self._random_string = str(uuid.uuid1())
        self._root_dir = Path(__file__).parent.parent / ".test_cache"
        self._root_dir.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=1)
    def get_auto_remove_tmp_dir(self):
        path = self._root_dir / self._random_string
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __del__(self):
        path = self.get_auto_remove_tmp_dir()
        shutil.rmtree(path)

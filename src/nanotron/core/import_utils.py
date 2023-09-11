import importlib
import importlib.metadata as importlib_metadata
import warnings
from typing import Tuple, Union


# https://github.com/huggingface/transformers/blob/f67dac97bdc63874f2288546b3fa87e69d2ea1c8/src/transformers/utils/import_utils.py#L41
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib_metadata.version(pkg_name)
            package_exists = True
        except importlib_metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def _can_import_from_module(module: str, name: str) -> bool:
    """
    Check if a specific module can be imported from a package.
    """
    if not _is_package_available(module):
        return False
    try:
        spec = importlib.util.find_spec(module)
        module_obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_obj)
        return hasattr(module_obj, name)
    except Exception as e:
        warnings.warn(f"Unable to import {name} from {module}: {e}")
        return False

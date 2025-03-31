# flake8: noqa
from nanotron.config.config import *
from nanotron.config.models_config import *
from nanotron.config.utils_config import *
from nanotron.config.lighteval_config import *

# Singleton for current config
_current_config = None


def set_config(config):
    """Set the current config"""
    global _current_config
    _current_config = config


def get_config():
    """Get the current config"""
    global _current_config
    return _current_config

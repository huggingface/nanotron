from dataclasses import dataclass
from typing import List, Optional, Union

import dacite
import pytest
from dacite import from_dict
from dacite.exceptions import UnionMatchError

from nanotron.config.config import explain_union_match_error


@dataclass
class _Nanoset:
    dataset_folder: Union[str, List[str]]
    dataset_weights: Optional[List[float]] = None


@dataclass
class _Pretrain:
    hf_dataset_or_datasets: Union[str, list, dict]
    hf_dataset_splits: Optional[Union[str, list]] = None


@dataclass
class _Data:
    dataset: Optional[Union[_Pretrain, _Nanoset]]
    seed: Optional[int] = 42


def _union_error(config_dict: dict) -> UnionMatchError:
    with pytest.raises(UnionMatchError) as excinfo:
        from_dict(data_class=_Data, data=config_dict, config=dacite.Config(cast=[], strict=True))
    return excinfo.value


def test_union_match_error_names_the_unknown_key():
    """strict=True makes one stray key fail every member of the union, and dacite
    reports only that nothing matched. The keys have to be named for the message
    to be actionable (#371)."""
    config_dict = {"dataset": {"dataset_folder": ["a", "b"], "dataset_weight": [0.5, 0.5]}}
    message = explain_union_match_error(_union_error(config_dict), config_dict)

    # The typo is named, against the type that otherwise fits.
    assert "_Nanoset: unknown keys: dataset_weight" in message
    # The member that cannot fit says what it is missing rather than staying silent.
    assert "missing required keys: hf_dataset_or_datasets" in message
    # dacite's own message is kept, so nothing is lost by wrapping it.
    assert "can not match type" in message


def test_union_match_error_falls_back_to_the_original_message():
    """A shape this does not understand degrades to dacite's message rather than
    raising out of the error handler."""

    class _Opaque(Exception):
        field_type = Union[str, int]
        value = "not a mapping"

        def __str__(self) -> str:
            return "original dacite message"

    assert explain_union_match_error(_Opaque(), {}) == "original dacite message"

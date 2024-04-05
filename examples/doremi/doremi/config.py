from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from nanotron.config import Config


@dataclass
class DoReMiArgs:
    smoothing_param: float = 1e-3
    step_size: float = 1.0

    domain_names: Optional[Union[str, List[str]]] = None
    domain_weights: Optional[Union[str, List[float]]] = None

    # NOTE: the path where you want to load the
    # reference model checkpoint for proxy training
    ref_model_resume_checkpoint_path: Optional[Path] = None

    def __post_init__(self):
        assert self.domain_names is not None, "Domain names must be provided."
        assert self.ref_model_resume_checkpoint_path is not None, "Reference model checkpoint path must be provided."

        self.domain_names = [str(name.strip()) for name in self.domain_names.split(",")]

        if self.domain_weights is not None:
            if isinstance(self.domain_weights, str):
                domain_weights = [float(weight.strip()) for weight in self.domain_weights.split(",")]
            else:
                domain_weights = self.domain_weights

            assert torch.allclose(
                torch.tensor(domain_weights).sum(), torch.tensor(1.0), rtol=1e-3
            ), "Domain weights must sum to 1.0."
            self.domain_weights = domain_weights

        self.ref_model_resume_checkpoint_path = Path(self.ref_model_resume_checkpoint_path)


@dataclass(kw_only=True)  # pylint: disable=unexpected-keyword-arg
class DoReMiConfig(Config):
    """Configuration for DoReMi's Proxy Training."""

    doremi: DoReMiArgs

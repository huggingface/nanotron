from dataclasses import dataclass, field
from typing import List, Optional

from nanotron.data.utils import parse_and_normalize_split


@dataclass
class NanosetConfig:
    """Configuration object for Nanoset datasets
    
    Attributes:
    
        random_seed (int): The seed for all RNG during dataset creation.

        sequence_length (int): The sequence length.

        data_path (str): Path to the .bin and .idx file without the extension

        split (Optional[str]): The split string, a comma separated weighting for the dataset splits
        when drawing samples from a single distribution. Not to be used with 'blend_per_split'.
        Defaults to None.

        split_vector Optional[List[float]]): The split string, parsed and normalized post-
        initialization. Not to be passed to the constructor.

        split_num_samples (list[int]): List containing the number of samples per split

        path_to_cache (str): Where all re-useable dataset indices are to be cached.
    """

    random_seed: int

    sequence_length: int
    
    split_num_samples: list[int]

    data_path: Optional[List[str]] = None

    split: Optional[str] = None

    split_vector: Optional[List[float]] = field(init=False, default=None)

    # TODO We add the cache because it need its somewhere, take a look later
    path_to_cache: str = None

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization. See
        https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        self.split_vector = parse_and_normalize_split(self.split)
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class LogItem:
    tag: str
    scalar_value: Union[float, int]
    log_format: Optional[str] = None

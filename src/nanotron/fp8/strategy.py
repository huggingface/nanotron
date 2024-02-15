from abc import ABC, abstractmethod


class ScalingStrategy(ABC):
    """Scaling stragey for dynamic quantization."""

    def __init__(self):
        pass

    @abstractmethod
    def __bool__(self):
        raise NotImplementedError


class WarmupStrategy(ScalingStrategy):
    pass


class DelayStrategy(ScalingStrategy):
    pass


class IntimeStrategy(ScalingStrategy):
    pass


class SkipOverflowStrategy(ScalingStrategy):
    pass


class SkipZeroOnlyStrategy(ScalingStrategy):
    pass

def post_init(cls):
    """Decorator to call __post_init__ method after __init__ method of a class."""
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if hasattr(self, "post_init"):
            self.__post_init__()

    cls.__init__ = new_init
    return cls

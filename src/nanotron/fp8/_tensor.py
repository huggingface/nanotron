import torch

from nanotron.fp8.meta import FP8Meta


# This is within core, the end user never have to look at this
class _WrapperTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        t, kwargs = cls.get_wrapper_properties(*args, **kwargs)
        if "size" not in kwargs:
            size = t.size()
        else:
            size = kwargs["size"]
            del kwargs["size"]
        if "dtype" not in kwargs:
            kwargs["dtype"] = t.dtype
        if "layout" not in kwargs:
            kwargs["layout"] = t.layout
        if "device" not in kwargs:
            kwargs["device"] = t.device
        if "requires_grad" not in kwargs:
            kwargs["requires_grad"] = False
        # Ignore memory_format and pin memory for now as I don't know how to
        # safely access them on a Tensor (if possible??)

        wrapper = torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)
        wrapper._validate_methods()
        return wrapper

    @classmethod
    def get_wrapper_properties(cls, *args, **kwargs):
        # Should return both an example Tensor and a dictionaly of kwargs
        # to override any of that example Tensor's properly.
        # This is very similar to the `t.new_*(args)` API
        raise NotImplementedError("You need to implement get_wrapper_properties")

    def _validate_methods(self):
        # Skip this if not in debug mode?
        # Changing these on the python side is wrong as it would not be properly reflected
        # on the c++ side
        # This doesn't catch attributes set in the __init__
        forbidden_overrides = ["size", "stride", "dtype", "layout", "device", "requires_grad"]
        for el in forbidden_overrides:
            if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
                raise RuntimeError(
                    f"Subclass {self.__class__.__name__} is overwriting the "
                    f"property {el} but this is not allowed as such change would "
                    "not be reflected to c++ callers."
                )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


from torch.utils._pytree import tree_map


class _FP8Tensor(_WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, diag):
        # return diag, {"size": diag.size() + diag.size()}
        return diag, {}

    def __init__(self, data: torch.Tensor, fp8_meta: FP8Meta):
        self._tensor = data

    @property
    def data(self):
        return self._tensor

    @data.setter
    def data(self, data):
        self._tensor = data

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return torch.diag(e._diag) if isinstance(e, _FP8Tensor) else e

        def wrap(e):
            return _FP8Tensor(torch.diag(e)) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        return rs


class FP8E4M3Tensor(_FP8Tensor):
    def __init__(self):
        pass

import torch
import torch.utils._pytree as pytree


class QuantTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            device=data.device,
        )

    # @torch._dynamo.disable
    def __init__(self, data: torch.Tensor):
        self._data = data

    def __tensor_flatten__(self):
        return ["_data"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["_data"], *tensor_attributes)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self._data})"

    # @classmethod
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     kwargs = kwargs or dict()

    #     if func is F.linear:
    #         return _BitNetTrainingLinear.apply(*args, **kwargs)

    #     with torch._C.DisableTorchFunctionSubclass():
    #         return func(*args, **kwargs)

    # adapted from FP8 implementation of WeightWithDynamicFloat8CastTensor
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        out = func(
            *pytree.tree_map_only(cls, lambda x: x._data, args),
            **pytree.tree_map_only(cls, lambda x: x._data, kwargs),
        )

        if func is torch.ops.aten.copy_.default:
            # return original object
            return args[0]
        elif func in {
            torch.ops.aten.t.default,
            torch.ops.aten.detach.default,
            torch.ops.aten.empty_like.default,
            torch.ops.aten.new_zeros.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.view.default,
            torch.ops.aten.as_strided.default,
            torch.ops.aten._to_copy.default,
            torch.ops.aten._pin_memory.default,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.clone.default,
        }:
            # return new wrapped object
            return pytree.tree_map_only(torch.Tensor, lambda x: cls(x), out)
        else:
            # return new unwrapped object
            return out


if __name__ == "__main__":
    tensor = torch.randn(2, 2, device="cuda")
    quant_tensor = QuantTensor(tensor)

    assert 1 == 1

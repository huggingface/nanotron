import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from functorch.dim import tree_map
from torch import nn

from nanotron import distributed as dist
from nanotron import logging
from nanotron.fp8.tensor import FP8Tensor

if TYPE_CHECKING:
    from nanotron.models import NanotronModel

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class SlicesPair:
    local_slices: Tuple[slice, ...]
    global_slices: Tuple[slice, ...]

    @staticmethod
    def slice_to_str(s: slice):
        # e.g. slice(0, 10, 2) -> "0,10,2"
        # e.g. slice(None, None, None) -> "None,None,None"
        return ",".join(str(x) if x is not None else "None" for x in (s.start, s.stop, s.step))

    @staticmethod
    def str_to_slice(s: str):
        return slice(*(int(x) if x != "None" else None for x in s.split(",")))

    def __str__(self):
        # e.g. local_slices (slice(0, 10, 2), slice(None, None, None)) -> "0,10,2|None,None,None"
        local_slices_str = "|".join(map(self.slice_to_str, self.local_slices))
        # e.g. global_slices (slice(0, 20, 4), slice(None, None, None)) -> "0,20,4|None,None,None"
        global_slices_str = "|".join(map(self.slice_to_str, self.global_slices))
        # e.g. "0,10,2|None,None,None#0,20,4|None,None,None"
        return f"{local_slices_str}#{global_slices_str}"

    @classmethod
    def from_str(cls, string: str):
        local_slices_str, global_slices_str = string.split("#")
        local_slices = tuple(map(cls.str_to_slice, local_slices_str.split("|")))
        global_slices = tuple(map(cls.str_to_slice, global_slices_str.split("|")))
        return cls(local_slices, global_slices)

    @classmethod
    def tuple_to_str(cls, pairs):
        # e.g. 2 SlicesPair, 1st SlicesPair local_slices "0,10,2|None,None,None" and global_slices "0,10,2|None,None,None"
        # 2nd SlicesPair local_slices "0,20,4|None,None,None" and global_slices "0,40,8|None,None,None"
        # -> "0,10,2|None,None,None#0,10,2|None,None,None;0,20,4|None,None,None#0,40,8|None,None,None"
        return ";".join(map(str, pairs))

    @classmethod
    def tuple_from_str(cls, string: str):
        return tuple(map(cls.from_str, string.split(";")))


@dataclasses.dataclass
class TiedInfo:
    name: str
    # name must be defined starting from `root_module` (e.g. root_module.dense0.dense1.weight)
    root_module: nn.Module
    global_ranks: Tuple[int, ...]
    # None signifies that we do not reduce
    reduce_op: Optional[dist.ReduceOp]

    def get_full_name_from_model(self, model: nn.Module) -> str:
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""
        return self.get_full_name_from_module_id_to_prefix(module_id_to_prefix)

    def get_full_name_from_module_id_to_prefix(self, module_id_to_prefix: Dict[int, str]) -> str:
        return f"{module_id_to_prefix[id(self.root_module)]}{self.name}"  # this assumes root_module is part of module_id_to_prefix


@dataclasses.dataclass
class ShardedInfo:
    global_ranks: Tuple[int, ...]
    # Info of to what slice of the unsharded tensor (global_slices) the current sharded tensor corresponds (local_slices)
    local_global_slices_pairs: Tuple[SlicesPair, ...]
    # The shape of the unsharded tensor
    unsharded_shape: Tuple[int, ...]

    def is_tp_sharded(self, parallel_context) -> bool:
        return set(dist.get_global_ranks(parallel_context.tp_pg)).issubset(set(self.global_ranks))

    def is_expert_sharded(self, parallel_context) -> bool:
        return set(dist.get_global_ranks(parallel_context.expert_pg)).issubset(set(self.global_ranks))

    def is_dp_sharded(self, parallel_context):
        return set(dist.get_global_ranks(parallel_context.dp_pg)).issubset(set(self.global_ranks))


# class NanotronParameter(nn.Parameter):
#     """Base class for all parameters in Nanotronmodels

#     A NanotronParameter can have specific properties:
#      - sharded: the parameter is considered to be `sharded` across multiple devices
#      - tied: the parameter is considered to be `tied` with other parameters. We sum gradients over those.

#     .. note::
#         Notes about tied weights:
#         - Tied weights means weights that need to be synced only within the same DP rank, regardless if they are part of TP strategy or just shared weights between two layers.
#         - Syncing tied weights usually require to sum gradients.
#         - Some weights are synced without needing to reduce grads over ranks. They can be in the same device (ex: enc/dec embeds in the same PP stage) or they can be duplicated across TP and duplicate the workload across TP ranks (ex: LN using traditional TP)
#         - Even if some weights don't need their grads to be reduced, it's still useful for them to be marked as tied. For example, current serialization format requires to mark them correctly.
#     """

#     NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME = "__nanotron_metadata__"
#     NANOTRON_PARAMETER_METADATA_TIED_KEY = "tied"
#     NANOTRON_PARAMETER_METADATA_SHARDED_KEY = "sharded"

#     def __new__(cls, tensor: torch.Tensor, requires_grad: bool = True):
#         assert tensor.data.is_floating_point() or tensor.data.requires_grad is False

#         # data = tensor.data.detach() if tensor.data.is_floating_point() else tensor.data
#         # requires_grad = requires_grad if data.is_floating_point() else False

#         if tensor.data.is_floating_point():
#             data = tensor.data.detach()
#         else:
#             # NOTE: FP8 tensor has int dtype, you can't .detach() an integer tensor!
#             requires_grad = False
#             data = tensor.data

#         # param = nn.Parameter.__new__(cls, data=data, requires_grad=requires_grad)
#         # param.data =
#         # NOTE: this somehow makes the param has the methods of NanotronParameter
#         param = nn.Parameter._make_subclass(cls, data=data)
#         param.requires_grad = requires_grad

#         if isinstance(tensor, NanotronParameter):
#             # Check that we don't inherit a weird class
#             # We copy in order not to make in-place operation
#             assert type(tensor) == NanotronParameter
#             setattr(
#                 param,
#                 cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME,
#                 getattr(tensor, cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME).copy(),
#             )
#         else:
#             setattr(param, cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME, {})

#         return param

#     def _set_metadata(self, key: str, value: Any):
#         metadata = getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)

#         if key in metadata:
#             raise ValueError(
#                 f"We shouldn't override previous metadata. Key to be overridden: {key}, current metadata: {metadata}"
#             )
#         else:
#             metadata[key] = value

#     def mark_as_tied(
#         self,
#         name: str,
#         global_ranks: Tuple[int, ...],
#         reduce_op: Optional[dist.ReduceOp],
#         root_module: "NanotronModel",
#     ):
#         self._set_metadata(
#             self.NANOTRON_PARAMETER_METADATA_TIED_KEY,
#             TiedInfo(name=name, global_ranks=global_ranks, reduce_op=reduce_op, root_module=root_module),
#         )

#     def get_tied_info(self) -> TiedInfo:
#         return getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)[
#             self.NANOTRON_PARAMETER_METADATA_TIED_KEY
#         ]

#     @property
#     def is_tied(self) -> bool:
#         return self.NANOTRON_PARAMETER_METADATA_TIED_KEY in getattr(
#             self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME
#         )

#     def mark_as_sharded(
#         self,
#         global_ranks: Tuple[int, ...],
#         local_global_slices_pairs: Tuple[SlicesPair, ...],
#         unsharded_shape: Tuple[int, ...],
#     ):
#         self._set_metadata(
#             self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY,
#             ShardedInfo(
#                 global_ranks=global_ranks,
#                 local_global_slices_pairs=local_global_slices_pairs,
#                 unsharded_shape=unsharded_shape,
#             ),
#         )

#     def get_sharded_info(self) -> ShardedInfo:
#         return getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)[
#             self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY
#         ]

#     @property
#     def is_sharded(self) -> bool:
#         return self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY in getattr(
#             self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME
#         )

#     def __repr__(self):
#         return f"NanotronParameter({super().__repr__()})"


class NanotronParameter(nn.Parameter):
    """Base class for all parameters in Nanotronmodels

    A NanotronParameter can have specific properties:
     - sharded: the parameter is considered to be `sharded` across multiple devices
     - tied: the parameter is considered to be `tied` with other parameters. We sum gradients over those.

    .. note::
        Notes about tied weights:
        - Tied weights means weights that need to be synced only within the same DP rank, regardless if they are part of TP strategy or just shared weights between two layers.
        - Syncing tied weights usually require to sum gradients.
        - Some weights are synced without needing to reduce grads over ranks. They can be in the same device (ex: enc/dec embeds in the same PP stage) or they can be duplicated across TP and duplicate the workload across TP ranks (ex: LN using traditional TP)
        - Even if some weights don't need their grads to be reduced, it's still useful for them to be marked as tied. For example, current serialization format requires to mark them correctly.
    """

    NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME = "__nanotron_metadata__"
    NANOTRON_PARAMETER_METADATA_TIED_KEY = "tied"
    NANOTRON_PARAMETER_METADATA_SHARDED_KEY = "sharded"

    def __new__(cls, tensor: torch.Tensor, requires_grad: bool = True):
        try:
            assert tensor.data.is_floating_point() or tensor.data.requires_grad is False
        except AttributeError:
            assert 1 == 1

        # data = tensor.data.detach() if tensor.data.is_floating_point() else tensor.data
        # requires_grad = requires_grad if data.is_floating_point() else False

        if tensor.data.is_floating_point():
            data = tensor.data.detach()
        else:
            # NOTE: FP8 tensor has int dtype, you can't .detach() an integer tensor!
            data = tensor.data

        # param = nn.Parameter.__new__(cls, data=data, requires_grad=requires_grad)
        # param.data =
        # NOTE: this somehow makes the param has the methods of NanotronParameter
        param = nn.Parameter._make_wrapper_subclass(
            cls,
            size=data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            device=data.device,
            requires_grad=data.requires_grad,
        )

        if isinstance(tensor, NanotronParameter):
            # Check that we don't inherit a weird class
            # We copy in order not to make in-place operation
            assert type(tensor) == NanotronParameter
            setattr(
                param,
                cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME,
                getattr(tensor, cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME).copy(),
            )
        else:
            setattr(param, cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME, {})

        return param

    def __init__(self, tensor: Union[torch.Tensor, FP8Tensor]):
        self._data = tensor

    def _set_metadata(self, key: str, value: Any):
        metadata = getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)

        if key in metadata:
            raise ValueError(
                f"We shouldn't override previous metadata. Key to be overridden: {key}, current metadata: {metadata}"
            )
        else:
            metadata[key] = value

    def mark_as_tied(
        self,
        name: str,
        global_ranks: Tuple[int, ...],
        reduce_op: Optional[dist.ReduceOp],
        root_module: "NanotronModel",
    ):
        self._set_metadata(
            self.NANOTRON_PARAMETER_METADATA_TIED_KEY,
            TiedInfo(name=name, global_ranks=global_ranks, reduce_op=reduce_op, root_module=root_module),
        )

    def get_tied_info(self) -> TiedInfo:
        return getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)[
            self.NANOTRON_PARAMETER_METADATA_TIED_KEY
        ]

    @property
    def is_tied(self) -> bool:
        return self.NANOTRON_PARAMETER_METADATA_TIED_KEY in getattr(
            self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME
        )

    def mark_as_sharded(
        self,
        global_ranks: Tuple[int, ...],
        local_global_slices_pairs: Tuple[SlicesPair, ...],
        unsharded_shape: Tuple[int, ...],
    ):
        self._set_metadata(
            self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY,
            ShardedInfo(
                global_ranks=global_ranks,
                local_global_slices_pairs=local_global_slices_pairs,
                unsharded_shape=unsharded_shape,
            ),
        )

    def get_sharded_info(self) -> ShardedInfo:
        return getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)[
            self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY
        ]

    @property
    def is_sharded(self) -> bool:
        return self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY in getattr(
            self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME
        )

    def __repr__(self):
        return f"NanotronParameter({super().__repr__()})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e._data if isinstance(e, NanotronParameter) else e

        # def wrap(e):
        #     # TODO(xrsrke): in some ops, the first metadata isn't related to the output
        #     metadatas = tuple(a._data.fp8_meta for a in args if isinstance(a, NanotronParameter))
        #     return cls(e, fp8_meta=metadatas[0]) if isinstance(e, torch.Tensor) else e

        def wrap(e):
            # return cls(e) if not isinstance(e, NanotronParameter) else e
            if not isinstance(e, NanotronParameter) and isinstance(e, (torch.Tensor, FP8Tensor)):
                return cls(e)
            else:
                return e

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        OPS_THAT_RETURN_ORIGINAL_TENSOR = [torch.ops.aten.native_layer_norm.default]

        if func == torch.ops.aten.detach.default:
            # NOTE: this is for parameter.data or parameter.detach()
            return args[0].data
        else:
            outputs = func(*args, **kwargs)

            if func in OPS_THAT_RETURN_ORIGINAL_TENSOR:
                return outputs
            else:
                # if len(outputs) == 1 and not isinstance(outputs, torch.Tensor):
                #     # NOTE: in some distributed operation, it doesn't return anything
                #     # but do in-place operation
                #     return outputs
                # else:
                #     return tree_map(wrap, outputs)
                return tree_map(wrap, outputs)


def sanity_check(root_module: nn.Module):
    """Makes sure that the module is in Nanotronformat

    Format:
     - all parameters are `NanotronParameter`, this allows us to add metadata to a parameter.
    """
    for name, param in root_module.named_parameters():
        if not isinstance(param, NanotronParameter):
            raise ValueError(
                f"Nanotronrequires model to be in Nanotronformat, ie all parameters are required to be a NanotronParameter. {name} isn't."
            )

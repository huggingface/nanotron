import dataclasses


@dataclasses.dataclass
class TensorPointer:
    """Dataclass specifying from which rank we need to query a tensor from in order to access data"""

    # Needed to understand from which rank to get the tensor
    # TODO @thomasw21: Maybe add which group it belongs to as well? Typically this is highly correlated to `p2p.pg`
    group_rank: int
    # TODO @thomasw21: Maybe add a tag (torch.distributed.send/recv allow for tagging)

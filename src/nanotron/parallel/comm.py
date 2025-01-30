from typing import Dict


class AsyncCommBucket:
    """

    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    RuntimeError: expected Variable or None (got tuple)
        Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    RuntimeError: expected Variable or None (got tuple)
    """

    _async_op: Dict[int, "dist.Work"] = {}

    @staticmethod
    def add(tensor_id: int, work: "dist.Work"):
        AsyncCommBucket._async_op[tensor_id] = work

    @staticmethod
    def get(tensor_id: int):
        return AsyncCommBucket._async_op.get(tensor_id)

    @staticmethod
    def wait(tensor_id: int):
        work = AsyncCommBucket._async_op.pop(tensor_id)
        work.wait()

import torch
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.parallel import ParallelContext


@rerun_if_address_is_in_use(max_try=2)
def test_rerun():
    # spawn(_test_rerun, tp=2, dp=1, pp=1, hello=1)
    init_distributed(tp=2, dp=1, pp=2)(_test_rerun)(hello=1)


def _test_rerun(tp: int, pp: int, dp: int, hello: int):
    parallel_context = ParallelContext(data_parallel_size=dp, pipeline_parallel_size=pp, tensor_parallel_size=tp)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # if torch.randint(0, 6, (1,)).item() < 4:
    #     raise Exception(f"Address already in use hello={hello}")

    parallel_context.destroy()

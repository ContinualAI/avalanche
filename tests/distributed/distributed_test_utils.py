import contextlib
import os

import torch

from avalanche.distributed import DistributedHelper


def common_dst_tests_setup():
    use_gpu_in_tests = os.environ.get("USE_GPU", "false").lower() in ["1", "true"]
    use_gpu_in_tests = use_gpu_in_tests and torch.cuda.is_available()
    DistributedHelper.init_distributed(1234, use_cuda=use_gpu_in_tests)
    return use_gpu_in_tests


def check_skip_distributed_test() -> bool:
    return os.environ.get("DISTRIBUTED_TESTS", "false").lower() not in ["1", "true"]


def check_skip_distributed_slow_test() -> bool:
    return check_skip_distributed_test() or os.environ.get(
        "FAST_TEST", "false"
    ).lower() in ["1", "true"]


@contextlib.contextmanager
def suppress_dst_tests_output():
    if os.environ["LOCAL_RANK"] != 0:
        with contextlib.redirect_stderr(None):
            with contextlib.redirect_stdout(None):
                yield
    else:
        yield


__all__ = [
    "common_dst_tests_setup",
    "check_skip_distributed_test",
    "check_skip_distributed_slow_test",
    "suppress_dst_tests_output",
]

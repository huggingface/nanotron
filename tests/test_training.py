"""Script to test correctness of training script by comparing loss value after 100th iteration with expected loss value

```bash
python tests/test_training.py
```
"""

import atexit
import os
import re
import signal
import subprocess
import time

EXPECTED_LOSS = 8e-03
CONFIG_FILE = "configs/config_correctness.yaml"
TRAIN_SCRIPT = "scripts/train.py"
NUM_GPUS = 8
CHECK_ITERATION = 100


def exit_with_children():
    """Kill all children processes when this process exits"""
    os.killpg(0, signal.SIGKILL)


def extract_loss(line):
    """Extract loss value from the line"""
    # extract loss value of the type | lm_loss: 7.087376E-03 | OR | lm_loss: 7.087376E+03 |
    try:
        return float(re.search(r"lm_loss: (\d+.\d+E[-+]?\d+)", line.decode("utf-8")).group(1))
    except AttributeError:
        raise ValueError(f"Could not extract loss value from line: {line}")


if __name__ == "__main__":
    cmd = f"USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node={NUM_GPUS} {TRAIN_SCRIPT} --config-file {CONFIG_FILE}"
    start_time = time.time()

    os.setpgrp()  # create new process group, become its leader
    atexit.register(exit_with_children)  # kill all children processes when this process exits

    # read logs in streaming fashion
    for line in subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout:
        print(line.decode("utf-8"), end="")

        # for all iterations >= 30, loss should be below 0.01
        if re.search(r"iteration: (\d+) / ", line.decode("utf-8")):
            if int(re.search(r"iteration: (\d+) / ", line.decode("utf-8")).group(1)) >= 30:
                loss = extract_loss(line)
                if loss > 2e-02:
                    print("=" * 10, "TEST FAILED", "=" * 10)
                    print(f"Loss after 30th iteration is {loss} which is bigger than expected loss 0.01")
                    print(f"Time taken: {time.time() - start_time}")
                    exit(1)

        if re.search(rf"iteration: {CHECK_ITERATION} / ", line.decode("utf-8")):
            loss = extract_loss(line)
            if loss > EXPECTED_LOSS:
                print("=" * 10, "TEST FAILED", "=" * 10)
                print(
                    f"Loss after {CHECK_ITERATION}th iteration is {loss} which is bigger than expected loss {EXPECTED_LOSS}"
                )
                print(f"Time taken: {time.time() - start_time:.2f}s")
                exit(1)
            else:
                print("=" * 10, "TEST PASSED", "=" * 10)
                print(
                    f"Loss after {CHECK_ITERATION}th iteration is {loss} which is smaller than expected loss {EXPECTED_LOSS}"
                )
                print(f"Time taken: {time.time() - start_time:.2f}s")
                exit(0)

# """Script to test correctness of training script by comparing loss value after 100th iteration with expected loss value

# ```bash
# pytest -sv tests/test_train_llama.py or python tests/test_train_llama.py
# ```
# """

import atexit
import os
import re
import signal
import subprocess

CONFIG_FILE = "examples/config_train_llama.yaml"
CREATE_CONFIG_FILE = "examples/config_train_llama.py"
TRAIN_SCRIPT = "run_train.py"
NUM_GPUS = 8

## 100+ steps: lm_loss < 3.5
## 200  steps: lm_loss < 3

EXPECTED_LOSS = 3.5
CHECK_ITERATION = 100

EXPECTED_LOSS_END = 3
CHECK_ITERATION_END = 200


def exit_with_children():
    """Kill all children processes when this process exits"""
    os.killpg(0, signal.SIGKILL)


def extract_loss(line):
    """Extract loss value from the line"""
    # extract loss value of the type | lm_loss: 5.33
    try:
        return float(re.search(r"lm_loss: (\d+.\d)", line.decode("utf-8")).group(1))
    except AttributeError:
        raise ValueError(f"Could not extract loss value from line: {line}")


def test_tiny_llama():
    cmd = f"python {CREATE_CONFIG_FILE}"
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    cmd = f'FI_PROVIDER="efa" CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node={NUM_GPUS} --rdzv_endpoint=localhost:29800  {TRAIN_SCRIPT} --config-file {CONFIG_FILE}'
    os.setpgrp()  # create new process group, become its leader
    atexit.register(exit_with_children)  # kill all children processes when this process exits

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Read and print output in real-time
    while True:
        line = process.stdout.readline()
        if process.poll() is not None and line == b"":
            break
        if line:
            print(line.decode("utf-8"), end="")
            # for all iterations >= CHECK_ITERATION, loss should be below EXPECTED_LOSS
            if re.search(r"iteration: (\d+) / ", line.decode("utf-8")):
                if int(re.search(r"iteration: (\d+) / ", line.decode("utf-8")).group(1)) >= CHECK_ITERATION:
                    loss = extract_loss(line)
                    assert loss < EXPECTED_LOSS

            if re.search(rf"iteration: {CHECK_ITERATION_END} / ", line.decode("utf-8")):
                loss = extract_loss(line)
                assert loss < EXPECTED_LOSS_END

    process.wait()  # Wait for the process to finish
    assert process.returncode == 0


if __name__ == "__main__":
    cmd = f"python {CREATE_CONFIG_FILE}"
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    cmd = f'FI_PROVIDER="efa" CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node={NUM_GPUS} --rdzv_endpoint=localhost:29800  {TRAIN_SCRIPT} --config-file {CONFIG_FILE}'
    os.setpgrp()  # create new process group, become its leader
    atexit.register(exit_with_children)  # kill all children processes when this process exits

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        # Read and print output in real-time
        while True:
            line = process.stdout.readline()
            if process.poll() is not None and line == b"":
                break
            if line:
                print(line.decode("utf-8"), end="")

                # for all iterations >= CHECK_ITERATION, loss should be below EXPECTED_LOSS
                if re.search(r"iteration: (\d+) / ", line.decode("utf-8")):
                    if int(re.search(r"iteration: (\d+) / ", line.decode("utf-8")).group(1)) >= CHECK_ITERATION:
                        loss = extract_loss(line)
                        assert loss < EXPECTED_LOSS
                # at iteration= CHECK_ITERATION, loss should be below EXPECTED_LOSS_END
                if re.search(rf"iteration: {CHECK_ITERATION_END} / ", line.decode("utf-8")):
                    loss = extract_loss(line)
                    assert loss < EXPECTED_LOSS_END
        process.wait()  # Wait for the process to finish
        assert process.returncode == 0
    except AssertionError:
        print("Command failed with exit code:", process.returncode)
    else:
        print("Command executed successfully.")

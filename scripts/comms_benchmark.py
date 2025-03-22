import time

import numpy as np
from mpi4py import MPI


def bandwith_test(size_in_mb=100, trials=10, warmup_trials=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    msg_size = size_in_mb * 1024 * 1024  # To convert into bytes
    tot_time = 0

    if size < 2:
        raise ValueError("The benchmark requires 2 MPI processes.")

    comm.Barrier()  # Synchronize before warmup

    for _ in range(warmup_trials):
        if rank == 0:
            data = np.ones(msg_size, dtype="b")  # Array of bytes
            comm.Send([data, MPI.BYTE], dest=1, tag=0)
            comm.Recv([data, MPI.BYTE], source=1, tag=1)
        elif rank == 1:
            data = np.empty(msg_size, dtype="b")
            comm.Recv([data, MPI.BYTE], source=0, tag=0)
            comm.Send([data, MPI.BYTE], dest=0, tag=1)

    comm.Barrier()  # Synchronize the processes before benchmark

    for _ in range(0, trials):
        if rank == 0:
            data = np.ones(msg_size, dtype="b")  # array of bytes
            start_time = time.time()
            comm.Send([data, MPI.BYTE], dest=1, tag=0)
            comm.Recv([data, MPI.BYTE], source=1, tag=1)
            end_time = time.time()
            elapsed_time = end_time - start_time
            tot_time += elapsed_time
        elif rank == 1:
            data = np.empty(msg_size, dtype="b")
            comm.Recv([data, MPI.BYTE], source=0, tag=0)
            comm.Send([data, MPI.BYTE], dest=0, tag=1)

    comm.Barrier()  # Synchronize the processes after benchmark

    if rank == 0:
        avg_time = tot_time / trials
        bandwidth = 2 * size_in_mb / avg_time  # Round trip
        print(f"Average Bandwidth: {bandwidth:.2f} MB/s")


def latency_test(num_trials=1000):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    msg_size = 1  # 1 byte
    tot_time = 0

    if size < 2:
        raise ValueError("The benchmark requires at least 2 MPI processes.")

    # Synchronize all processes before starting the benchmark
    comm.Barrier()

    for _ in range(num_trials):
        if rank == 0:
            data = np.ones(msg_size, dtype="b")
            start_time = time.perf_counter()
            comm.Send([data, MPI.BYTE], dest=1, tag=0)
            comm.Recv([data, MPI.BYTE], source=1, tag=1)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            tot_time += elapsed_time
        elif rank == 1:
            data = np.empty(msg_size, dtype="b")
            comm.Recv([data, MPI.BYTE], source=0, tag=0)
            comm.Send([data, MPI.BYTE], dest=0, tag=1)

    # Synchronize all processes after finishing the benchmark
    comm.Barrier()

    if rank == 0:
        avg_latency = (tot_time / num_trials) * 1e6  # Convert to microseconds
        print(f"Average Latency: {avg_latency:.2f} µs")


if __name__ == "__main__":
    bandwith_test()
    latency_test()

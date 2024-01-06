# Copyright 2021 HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by pipegoose's contributors.

from abc import ABC, abstractclassmethod
from typing import TypedDict

from nanotron.distributed.parallel_mode import ParallelMode
from torch.distributed import ProcessGroup


class ProcessGroupResult(TypedDict):
    local_rank: int
    local_world_size: int
    process_group: ProcessGroup
    parallel_mode: ParallelMode


class ProcessGroupInitializer(ABC):
    """A base class for initializing process groups in 3D parallelism."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size

    @abstractclassmethod
    def init_dist_group(self) -> ProcessGroupResult:
        raise NotImplementedError

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""
mlperf dlrm_v3 inference benchmarking tool.
"""

import contextlib
import logging
import os
from typing import Callable, Dict, List, Optional

import gin

import torch

from modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torch.profiler import profile, profiler, ProfilerActivity  # pyre-ignore [21]
from torch.utils.tensorboard import SummaryWriter
from torchrec.metrics.auc import AUCMetricComputation
# from torchrec.metrics.ndcg import NDCGComputation
# from torchrec.metrics.recall import RecallMetricComputation
# from torchrec.metrics.auprc import AUPRCMetricComputation

from torchrec.metrics.mae import MAEMetricComputation
from torchrec.metrics.mse import MSEMetricComputation
# from torchrec.metrics.ne import NEMetricComputation
from torchrec.metrics.accuracy import AccuracyMetricComputation

from torchrec.metrics.rec_metric import RecMetricComputation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils")


# def _on_trace_ready_fn(
#     rank: Optional[int] = None,
# ) -> Callable[[torch.profiler.profile], None]:
#     def handle_fn(p: torch.profiler.profile) -> None:
#         bucket_name = "hammer_gpu_traces"
#         pid = os.getpid()
#         rank_str = f"_rank_{rank}" if rank is not None else ""
#         file_name = f"libkineto_activities_{pid}_{rank_str}.json"
#         manifold_path = "tree/dlrm_v3_bench"
#         target_object_name = manifold_path + "/" + file_name + ".gz"
#         path = f"manifold://{bucket_name}/{manifold_path}/{file_name}"
#         logger.warning(
#             p.key_averages(group_by_input_shape=True).table(
#                 sort_by="self_cuda_time_total"
#             )
#         )
#         logger.warning(
#             f"trace url: https://www.internalfb.com/intern/perfdoctor/trace_view?filepath={target_object_name}&bucket={bucket_name}"
#         )
#         p.export_chrome_trace(path)

#     return handle_fn


def _on_trace_ready_fn(
    rank: Optional[int] = None,
) -> Callable[[torch.profiler.profile], None]:
    def handle_fn(p: torch.profiler.profile) -> None:
        pid = os.getpid()
        rank_str = f"_rank_{rank}" if rank is not None else ""
        file_name = f"libkineto_activities_{pid}_{rank_str}.json"
        # Save to local
        path = f"/tmp/{file_name}"
        logger.warning(
            p.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total"
            )
        )
        logger.warning(f"trace saved to: {path}")
        p.export_chrome_trace(path)

    return handle_fn


def profiler_or_nullcontext(enabled: bool, with_stack: bool):
    return (
        profile(
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=_on_trace_ready_fn(),
            with_stack=with_stack,
        )
        if enabled
        else contextlib.nullcontext()
    )


class Profiler:
    def __init__(self, rank, active: int = 50) -> None:
        self.rank = rank
        self._profiler: profiler.profile = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=10,
                warmup=20,
                active=active,
                repeat=1,
            ),
            on_trace_ready=_on_trace_ready_fn(self.rank),
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
        )

    def step(self) -> None:
        self._profiler.step()


@gin.configurable
class MetricsLogger:
    def __init__(
        self,
        multitask_configs: List[TaskConfig],
        batch_size: int,
        window_size: int,
        device: torch.device,
        rank: int,
        exp_name: str = None,  # default exp name
        default_tensorboard_log_path: str = os.environ.get('TENSORBOARD_LOG_DIR', '.'),  # default tb path in job
    ) -> None:
        self.rank: int = rank
        self.multitask_configs: List[TaskConfig] = multitask_configs
        all_classification_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type != MultitaskTaskType.REGRESSION
        ]
        all_regression_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type == MultitaskTaskType.REGRESSION
        ]
        assert all_classification_tasks + all_regression_tasks == [
            task.task_name for task in multitask_configs
        ]
        self.task_names: List[str] = all_classification_tasks + all_regression_tasks

        self.class_metrics: List[RecMetricComputation] = []
        if all_classification_tasks:
            # self.class_metrics.append(
            #     NEMetricComputation(
            #         my_rank=rank,
            #         batch_size=batch_size,
            #         n_tasks=len(all_classification_tasks),
            #         window_size=window_size,
            #     ).to(device)
            # )
            self.class_metrics.append(
                AUCMetricComputation(
                    my_rank=rank,
                    batch_size=batch_size,
                    n_tasks=len(all_classification_tasks),
                    window_size=window_size,
                ).to(device)
            )
            # self.class_metrics.append(
            #     NDCGComputation(
            #         my_rank=rank,
            #         batch_size=batch_size,
            #         n_tasks=len(all_classification_tasks),
            #         window_size=window_size,
            #         k=10,
            #     ).to(device)
            # )
            # self.class_metrics.append(
            #     RecallMetricComputation(
            #         my_rank=rank,
            #         batch_size=batch_size,
            #         n_tasks=len(all_classification_tasks),
            #         window_size=window_size,
            #     ).to(device)
            # )
            # self.class_metrics.append(
            #     AUPRCMetricComputation(
            #         my_rank=rank,
            #         batch_size=batch_size,
            #         n_tasks=len(all_classification_tasks),
            #         window_size=window_size,
            #     ).to(device)
            # )

            self.class_metrics.append(
                AccuracyMetricComputation(
                    my_rank=rank,
                    batch_size=batch_size,
                    n_tasks=len(all_classification_tasks),
                    window_size=window_size,
                ).to(device)
            )

        self.regression_metrics: List[RecMetricComputation] = []

        if all_regression_tasks:
            self.regression_metrics.append(
                MSEMetricComputation(
                    my_rank=rank,
                    batch_size=batch_size,
                    n_tasks=len(all_regression_tasks),
                    window_size=window_size,
                ).to(device)
            )
            self.regression_metrics.append(
                MAEMetricComputation(
                    my_rank=rank,
                    batch_size=batch_size,
                    n_tasks=len(all_regression_tasks),
                    window_size=window_size,
                ).to(device)
            )

        self.global_step: int = 0
        self.tb_logger: Optional[SummaryWriter] = None
        tensorboard_log_path = default_tensorboard_log_path + f"/{exp_name}" if exp_name else default_tensorboard_log_path
        if tensorboard_log_path != "":
            self.tb_logger = SummaryWriter(log_dir=tensorboard_log_path)

    @property
    def all_metrics(self) -> List[RecMetricComputation]:
        return self.class_metrics + self.regression_metrics

    def update(
        self, predictions: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor
    ) -> None:

        for metric in self.all_metrics:
            metric.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
            )
        self.global_step += 1

    def compute(self) -> Dict[str, float]:
        all_computed_metrics = {}

        for metric in self.all_metrics:
            computed_metrics = metric.compute()
            for computed in computed_metrics:
                all_values = computed.value.cpu()
                for i, task_name in enumerate(self.task_names):
                    key = f"metric/{str(computed.metric_prefix) + str(computed.name)}/{task_name}"
                    all_computed_metrics[key] = all_values[i]
        if self.rank == 0:
            logger.info(f"Step {self.global_step} metrics: {all_computed_metrics}")
        return all_computed_metrics

    def compute_and_log(
        self, additional_logs: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, float]:
        assert self.tb_logger is not None
        all_computed_metrics = self.compute()
        for k, v in all_computed_metrics.items():
            self.tb_logger.add_scalar(  # pyre-ignore [16]
                k,
                v,
                global_step=self.global_step,
            )

        # if additional_logs is not None:
        #     import torch.distributed as dist
        #     for tag, data in additional_logs.items():
        #         for data_name, data_value in data.items():
        #             if dist.is_initialized():
        #                 dist.all_reduce(data_value, op=dist.ReduceOp.SUM)
        #                 data_value /= dist.get_world_size()
        #             if self.rank == 0:
        #                 self.tb_logger.add_scalar(
        #                     f"{tag}/{data_name}",
        #                     data_value.detach().clone().cpu(),
        #                     global_step=self.global_step,
        #                 )

        # only log loss on rank 0
        if additional_logs is not None and self.rank == 0:
            for tag, data in additional_logs.items():
                for data_name, data_value in data.items():
                    self.tb_logger.add_scalar(
                        f"{tag}/{data_name}",
                        data_value.detach().clone().cpu(),
                        global_step=self.global_step,
                    )
        return all_computed_metrics

    def reset(self):
        for metric in self.all_metrics:
            metric.reset()


# the datasets we support
SUPPORTED_DATASETS = [
    "msan-iterable",
]


def get_dataset(name: str, new_path_prefix: str = ""):
    assert name in SUPPORTED_DATASETS, f"dataset {name} not supported"

    if name == "msan-iterable":
        from dlrm_v3.datasets.msan import DLRMv3MSANIterableDataset

        return (
            DLRMv3MSANIterableDataset,
            {
                # "seq_logs_file": 'shares/Bingads.algo.prod.IntentMatching/Data/users/shenqian/MSAN/L1Ranking/Data/DeepFBS_V3_20250416_20250422_20X/training_prepare/Behaviors.tsv',
                "seq_logs_file": 'Behaviors.tsv',  # Update to local path
                "max_lines": None,  # Set to None when processing all data
                "rank": 0,  # Will be set in make_train_test_dataloaders
                "world_size": 1,  # Will be set in make_train_test_dataloaders
            },
        )
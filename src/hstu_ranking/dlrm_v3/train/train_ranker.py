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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

import torch.distributed as dist

# pyre-strict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
import os
import sys
import time

import traceback

import gin
import torch
from dlrm_v3.checkpoint import load_dmp_checkpoint
from dlrm_v3.train.utils import (
    cleanup,
    eval_loop,
    make_model,
    make_optimizer_and_shard,
    make_train_test_dataloaders,
    setup,
    train_loop,
)
from dlrm_v3.utils import MetricsLogger
from torch import multiprocessing as mp
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger(__name__)


SUPPORTED_CONFIGS = {
    "msan-iterable": "msan_iterable.gin",
}

def _main_func(
    rank: int,
    world_size: int,
    master_port: int,
    gin_file: str,
    mode: str,
    dataset: str,
    data_path: str = None,
    exp_name: str = None,
) -> None:
    device = torch.device(f"cuda:{rank}")
    logger.info(f"rank: {rank}, world_size: {world_size}, device: {device}")
    
    # Log data path if provided
    if data_path:
        logger.info(f"Using data path: {data_path}")
    
    setup(
        rank=rank,
        world_size=world_size,
        master_port=master_port,
        device=device,
    )
    # parse all arguments
    gin.parse_config_file(gin_file)

    
    # Override gin config with command line data_path if provided
    if data_path:
        gin.bind_parameter('make_train_test_dataloaders.data_path', data_path)
        
        # ----- delete this part if your embedding file is not in the data_path -----
        # Bind embedding file paths if data_path is provided
        import os
        embedding_paths = {
            "event_id": os.path.join(data_path, "EventEmb.npy")
        }
        gin.bind_parameter('make_model.embedding_file_paths', embedding_paths)
        logger.info(f"Using embedding path: {embedding_paths['event_id']}")
        # ------- and specify your embedding path in the gin file --------

    model, model_configs, embedding_table_configs = make_model()

    # Count model parameters before sharding
    if rank == 0:  # Only log from rank 0 to avoid duplicate logs
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters (before sharding): {total_params:,}")
        logger.info(f"Trainable parameters (before sharding): {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

    model, optimizer = make_optimizer_and_shard(model=model, device=device)
    # load_dmp_checkpoint(model, optimizer)
    train_dataloader, test_dataloader = make_train_test_dataloaders(
        hstu_config=model_configs,
        embedding_table_configs=embedding_table_configs,
        rank=rank,
        world_size=world_size,
        data_path=data_path,  # Directly pass data path parameter
    )

    if exp_name is None:
        exp_name = dataset
    
    train_metrics = MetricsLogger(
        multitask_configs=model_configs.multitask_configs,
        batch_size=train_dataloader.batch_size,
        window_size=1000 if mode == "train" else sys.maxsize,
        device=device,
        rank=rank,
        exp_name=exp_name
    )

    eval_metrics = MetricsLogger(
        multitask_configs=model_configs.multitask_configs,
        batch_size=test_dataloader.batch_size,
        window_size=sys.maxsize,
        device=device,
        rank=rank,
        exp_name=exp_name.replace("train", "eval", 1)
    )

    
    # train loop
    try:
        if mode == "train":
            # First perform training
            train_loop(
                rank=rank,
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=test_dataloader,
                optimizer=optimizer,
                train_metric_logger=train_metrics,
                eval_metric_logger=eval_metrics,
                device=device,
                dataset=dataset,
            )

            
        elif mode == "eval":
            load_dmp_checkpoint(model, optimizer)


            eval_loop(
                rank=rank,
                model=model,
                dataloader=test_dataloader,
                # dataloader=train_dataloader,
                metric_logger=eval_metrics,
                device=device,
            )
    except Exception as e:
        logger.info(traceback.format_exc())
        # Clean up distributed environment
        cleanup()
        raise Exception(e)


def get_args():  # pyre-ignore [3]
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="debug", choices=SUPPORTED_CONFIGS.keys(), help="dataset"
    )
    parser.add_argument(
        "--mode", default="train", choices=["train", "eval"], help="mode"
    )
    parser.add_argument(
        "--world_size", type=int, default=None, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to the data directory (mount point)"
    )
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Experiment name for logging"
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


def main() -> None:
    args = get_args()
    logger.info(args)
    assert args.dataset in SUPPORTED_CONFIGS, f"Unsupported dataset: {args.dataset}"
    assert args.mode in ["train", "eval"], f"Unsupported mode: {args.mode}"
    
    # Automatically detect available GPU count
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    # Prioritize command line arguments, then environment variables, finally default to auto-detected GPU count
    WORLD_SIZE = args.world_size or int(os.environ.get("WORLD_SIZE", available_gpus))
    
    # Set NCCL environment variables before multiprocessing startup
    os.environ["NCCL_IB_DISABLE"] = "1"                    # Disable InfiniBand since Azure ML doesn't support it
    os.environ["NCCL_DEBUG"] = "WARN"                      # Reduce log output
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"    # Enable async error handling
    os.environ["NCCL_TIMEOUT"] = "600"                     # 10 minutes timeout
    
    # Ensure CUDA device initialization
    if torch.cuda.is_available():
        torch.cuda.init()
        logger.info(f"CUDA initialized with {torch.cuda.device_count()} devices")
    
    # Ensure not to exceed available GPU count
    if WORLD_SIZE > available_gpus:
        print(f"Warning: Requested {WORLD_SIZE} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        WORLD_SIZE = available_gpus
    
    # Use at least 1 GPU
    if WORLD_SIZE < 1:
        print("Warning: No GPUs available or WORLD_SIZE < 1. Using 1 GPU.")
        WORLD_SIZE = 1
    
    MASTER_PORT = str(get_free_port())
    gin_path = f"{os.path.dirname(__file__)}/gin/{SUPPORTED_CONFIGS[args.dataset]}"

    start_time = time.time()
    print(f"Starting train_ranker: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Using {WORLD_SIZE} GPUs for training")
    
    mp.start_processes(
        _main_func,
        args=(WORLD_SIZE, MASTER_PORT, gin_path, args.mode, args.dataset, args.data_path, args.exp_name),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total runtime: {duration:.2f} seconds ({duration/60:.2f} minutes)")


if __name__ == "__main__":
    main()

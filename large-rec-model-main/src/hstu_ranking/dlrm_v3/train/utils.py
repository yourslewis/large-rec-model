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

from torch.utils.tensorboard import SummaryWriter
import warnings

# pyre-strict
import logging
import os
import time
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import gin

import torch
import torchrec
from common import HammerKernel
from dlrm_v3.checkpoint import save_dmp_checkpoint
from dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from dlrm_v3.datasets.dataset import collate_fn, Dataset
from dlrm_v3.utils import get_dataset, MetricsLogger, Profiler
from modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim.optimizer import Optimizer

from torch.utils.data import DataLoader, Dataset as TorchDataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ShardedTensor
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)
# Suppress FBGEMM autograd warnings
warnings.filterwarnings("ignore", message=".*fbgemm.*autograd kernel was not registered.*")

TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}



def track_training_health(model: torch.nn.Module, optimizer: Optimizer, loss: torch.Tensor, step: int, rank: int = 0) -> None:
    """Track training health indicators"""
    if rank != 0:  # Only track on rank 0 to avoid duplicate logs
        return
    
    # Initialize tracking variables as function attributes
    if not hasattr(track_training_health, 'loss_history'):
        track_training_health.loss_history = []
        track_training_health.param_history = {}
        track_training_health.gradient_history = []
        
    track_training_health.loss_history.append(loss.item())
    
    # 1. Track loss trends (every 10 steps)
    if step % 10 == 0 and len(track_training_health.loss_history) >= 10:
        recent_losses = track_training_health.loss_history[-10:]
        avg_recent = sum(recent_losses) / len(recent_losses)
        
        if len(track_training_health.loss_history) >= 20:
            older_losses = track_training_health.loss_history[-20:-10]
            avg_older = sum(older_losses) / len(older_losses)
            
            improvement = (avg_older - avg_recent) / max(avg_older, 1e-8) * 100
            if improvement < 1:  # Less than 1% improvement
                logger.warning(f"Step {step}: Loss not decreasing significantly. Recent avg: {avg_recent:.6f}, Older avg: {avg_older:.6f}")
            else:
                logger.info(f"Step {step}: Loss improving {improvement:.2f}%. Recent avg: {avg_recent:.6f}")
    
    # 2. Track gradient norms
    total_norm = 0
    param_count = 0
    zero_grad_params = []
    large_grad_params = []
    no_grad_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Debug: Log detailed gradient info for first few steps
            if step < 5:
                logger.info(f"Step {step} - {name}: grad_norm={param_norm.item():.8f}, requires_grad={param.requires_grad}, shape={param.shape}")
            
            # Check for problematic gradients
            if param_norm.item() == 0:
                zero_grad_params.append(name)
            elif param_norm.item() > 10:  # Threshold for large gradients
                large_grad_params.append((name, param_norm.item()))
        else:
            no_grad_params.append(name)
            # Debug: Log parameters without gradients
            if step < 5:
                logger.warning(f"Step {step} - {name}: NO GRADIENT, requires_grad={param.requires_grad}, shape={param.shape}")
    
    total_norm = total_norm ** (1. / 2)
    track_training_health.gradient_history.append(total_norm)
    
    # Log gradient health (every 10 steps)
    if step % 10 == 0:
        logger.info(f"Step {step}: Gradient norm: {total_norm:.6f}, Params with gradients: {param_count}")
        
        if no_grad_params:
            logger.error(f"Step {step}: {len(no_grad_params)} params with NO gradients: {no_grad_params[:5]}...")
            
        if zero_grad_params:
            logger.warning(f"Step {step}: {len(zero_grad_params)} params with zero gradients: {zero_grad_params[:5]}...")
            
        if large_grad_params:
            logger.warning(f"Step {step}: {len(large_grad_params)} params with large gradients (>10)")
    
    # 3. Check for frozen parameters (first step only)
    if step == 0:
        frozen_params = []
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        logger.info(f"Model parameter status:")
        logger.info(f"  Trainable parameters: {len(trainable_params)}")
        logger.info(f"  Frozen parameters: {len(frozen_params)}")
        
        if frozen_params:
            logger.warning(f"  Frozen params (first 5): {frozen_params[:5]}")
    
    # 4. Track parameter updates (every 50 steps)
    if step % 50 == 0 and step > 0:
        param_changes = {}
        for name, param in model.named_parameters():
            if name not in track_training_health.param_history:
                track_training_health.param_history[name] = param.data.clone()
            else:
                old_param = track_training_health.param_history[name]
                param_change = (param.data - old_param).norm().item()
                param_norm = param.data.norm().item()
                relative_change = param_change / max(param_norm, 1e-8)
                param_changes[name] = relative_change
                track_training_health.param_history[name] = param.data.clone()
        
        # Find parameters with minimal updates
        if param_changes:
            stagnant_params = [(name, change) for name, change in param_changes.items() if change < 1e-6]
            if stagnant_params and len(stagnant_params) < 10:  # Don't spam if too many
                logger.warning(f"Step {step}: {len(stagnant_params)} params with minimal updates (<1e-6)")
            
            # Log parameter update summary
            avg_change = sum(param_changes.values()) / len(param_changes)
            max_change = max(param_changes.values())
            min_change = min(param_changes.values())
            logger.info(f"Step {step}: Param updates - Avg: {avg_change:.8f}, Max: {max_change:.8f}, Min: {min_change:.8f}")
    
    # 5. Anomaly detection
    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"Step {step}: NaN or Inf loss detected: {loss}")
        raise ValueError(f"Invalid loss value at step {step}")
    
    if total_norm > 1000:
        logger.error(f"Step {step}: Gradient explosion detected, norm: {total_norm}")
    elif total_norm < 1e-8:
        if param_count == 0:
            logger.error(f"Step {step}: NO GRADIENTS - Model parameters are frozen or not connected to loss!")
        else:
            logger.warning(f"Step {step}: Very small gradients detected, norm: {total_norm}")
    
    # 6. Track gradient trends (every 50 steps)
    if step % 50 == 0 and len(track_training_health.gradient_history) >= 10:
        recent_grads = track_training_health.gradient_history[-10:]
        avg_recent_grad = sum(recent_grads) / len(recent_grads)
        logger.info(f"Step {step}: Recent gradient norm average: {avg_recent_grad:.6f}")


def setup(
    rank: int, world_size: int, master_port: int, device: torch.device
) -> dist.ProcessGroup:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    BACKEND = dist.Backend.NCCL
    TIMEOUT = 600

    # initialize the process group
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=TIMEOUT))

    pg = dist.new_group(
        backend=BACKEND,
        timeout=timedelta(seconds=TIMEOUT),
    )

    # set device
    torch.cuda.set_device(device)

    # test_communication(rank, device, world_size)

    return pg

def test_communication(rank: int, device: torch.device, world_size: int) -> None:
    """Test distributed communication between all ranks"""
    try:
        # Test 1: Simple all_reduce
        test_tensor = torch.tensor([rank], dtype=torch.float32, device=device)
        dist.all_reduce(test_tensor)
        expected_sum = sum(range(world_size))  # 0+1+2+...+(world_size-1)
        
        if rank == 0:
            print(f"=== Communication Test Results ===")
            print(f"All-reduce test: {test_tensor.item():.0f} (expected: {expected_sum})")
        
        # Test 2: Broadcast test
        if rank == 0:
            broadcast_tensor = torch.tensor([42.0], device=device)
        else:
            broadcast_tensor = torch.tensor([0.0], device=device)
        
        dist.broadcast(broadcast_tensor, src=0)
        
        # Test 3: Barrier synchronization
        dist.barrier()
        
        if rank == 0:
            print(f"Broadcast test: {broadcast_tensor.item():.0f} (expected: 42)")
            print(f"All {world_size} ranks communication: ✅ SUCCESS")
            print("==================================")
        
        # Verify results
        if abs(test_tensor.item() - expected_sum) > 1e-6:
            raise RuntimeError(f"Rank {rank}: All-reduce failed, got {test_tensor.item()}, expected {expected_sum}")
        
        if abs(broadcast_tensor.item() - 42.0) > 1e-6:
            raise RuntimeError(f"Rank {rank}: Broadcast failed, got {broadcast_tensor.item()}, expected 42.0")
            
    except Exception as e:
        logger.error(f"Rank {rank}: Communication test FAILED: {e}")
        raise RuntimeError(f"Distributed communication setup failed on rank {rank}")

def test_communication_in_training(rank: int, device: torch.device) -> None:
    """Lightweight communication test during training"""
    try:
        test_tensor = torch.tensor([rank], dtype=torch.float32, device=device)
        dist.all_reduce(test_tensor)
        if rank == 0:
            logger.info(f"Step communication check: ✅ (sum={test_tensor.item():.0f})")
    except Exception as e:
        logger.error(f"Rank {rank}: Communication check failed: {e}")


def cleanup() -> None:
    dist.destroy_process_group()


class HammerToTorchDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        self.dataset: Dataset = dataset

    def __getitem__(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        self.dataset.load_query_samples([idx])
        sample = self.dataset.get_sample(idx)
        self.dataset.unload_query_samples([idx])
        return sample

    def __getitems__(
        self, indices: List[int]
    ) -> List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]]:
        self.dataset.load_query_samples(indices)
        samples = [self.dataset.get_sample(i) for i in indices]
        self.dataset.unload_query_samples(indices)
        return samples


# def check_frozen_params(model: torch.nn.Module, rank: int = 0) -> None:
#     """Check which parameters are frozen/trainable in the model"""
#     if rank == 0:  # Only print from rank 0 to avoid duplicate logs
#         logger.info("=== Parameter Freeze Status ===")
#         trainable_count = 0
#         frozen_count = 0
        
#         for name, param in model.named_parameters():
#             status = "TRAINABLE" if param.requires_grad else "FROZEN"
#             if 'embedding' in name.lower() or 'emb' in name.lower():
#                 logger.info(f"{name}: {status} (shape: {param.shape})")
            
#             if param.requires_grad:
#                 trainable_count += param.numel()
#             else:
#                 frozen_count += param.numel()
        
#         total_params = trainable_count + frozen_count
#         logger.info(f"Total parameters: {total_params:,}")
#         logger.info(f"Trainable parameters: {trainable_count:,} ({trainable_count/total_params*100:.1f}%)")
#         logger.info(f"Frozen parameters: {frozen_count:,} ({frozen_count/total_params*100:.1f}%)")
#         logger.info("==============================")

@gin.configurable
def make_model(
    dataset: str,
    embedding_file_paths: Optional[Dict[str, str]] = None,
) -> Tuple[torch.nn.Module, DlrmHSTUConfig, Dict[str, EmbeddingConfig]]:

    hstu_config = get_hstu_configs(dataset)
    table_config = get_embedding_table_config(dataset)

    if embedding_file_paths:
        # Use precomputed embeddings
        # logger.info("Using precomputed embeddings for HSTU model")

        # Avoid duplicate logging in distributed setup
        try:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info("Using precomputed embeddings for HSTU model")
        except:
            logger.info("Using precomputed embeddings for HSTU model")
        
        # Pass embedding file paths to the model
        model = DlrmHSTU(
            hstu_configs=hstu_config,
            embedding_tables=table_config,
            is_inference=False,
            embedding_file_paths=embedding_file_paths,
        )
    else:
        # Use standard HSTU configs
        logger.info("Using random initialization for HSTU model")        
        model = DlrmHSTU(
            hstu_configs=hstu_config,
            embedding_tables=table_config,
            is_inference=False,
        )
    
    model.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
    # check_frozen_params(model)
    # breakpoint()  # Debugging point to inspect model structure if needed

    return (
        model,
        hstu_config,
        table_config,
    )


@gin.configurable()
def dense_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    elif optimizer_name == "SGD":
        optimizer_cls = torch.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory


@gin.configurable()
def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "SGD":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "RowWiseAdagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory


def make_optimizer_and_shard(
    model: torch.nn.Module,
    device: torch.device,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    dense_opt_cls, dense_opt_args, dense_opt_factory = (
        dense_optimizer_factory_and_class()
    )

    sparse_opt_cls, sparse_opt_args, sparse_opt_factory = (
        sparse_optimizer_factory_and_class()
    )
    # Fuse sparse optimizer to backward step
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )

    # Shard model
    model = DistributedModelParallel(
        module=model,
        device=device,
    )

    # Create keyed optimizer
    all_optimizers = []
    all_params = {}
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
            else:
                all_params[k] = v

    if non_fused_sparse_params:
        all_optimizers.append(
            (
                "sparse_non_fused",
                KeyedOptimizerWrapper(
                    params=non_fused_sparse_params, optim_factory=sparse_opt_factory
                ),
            )
        )

    if all_params:
        all_optimizers.append(
            (
                "dense",
                KeyedOptimizerWrapper(
                    params=all_params,
                    optim_factory=dense_opt_factory,
                ),
            )
        )
    output_optimizer = CombinedOptimizer(all_optimizers)
    output_optimizer.init_state(set(model.sparse_grad_parameter_names()))
    return model, output_optimizer


@gin.configurable
def make_train_test_dataloaders(
    batch_size: int,
    dataset_type: str,
    hstu_config: DlrmHSTUConfig,
    train_split_percentage: float,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    new_path_prefix: str = "",
    num_workers: int = 0,  # Default value, will be automatically adjusted based on GPU count
    prefetch_factor: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,  # Default single GPU, actual value will be passed from train_ranker
    data_path: str = None,  # Data path parameter
    total_lines: int = 5411079,  # Default total lines for msan datasets
) -> Tuple[DataLoader, DataLoader]:
    dataset_class, kwargs = get_dataset(dataset_type, new_path_prefix)
    kwargs["embedding_config"] = embedding_table_configs
    
    # If data path is provided, pass it to the dataset
    if data_path:
        kwargs["mount_path"] = data_path
    
    
    if issubclass(dataset_class, IterableDataset):
        # For iterable datasets, pass rank and world_size
        kwargs["rank"] = rank
        kwargs["world_size"] = world_size
        
        # Create train and eval datasets with different parameters
        train_kwargs = kwargs.copy()
        eval_kwargs = kwargs.copy()
        
        train_lines = int(train_split_percentage * total_lines)
        eval_lines = total_lines - train_lines
        
        # For train dataset: read from beginning to train_split_percentage
        train_kwargs["max_lines"] = train_lines
        train_kwargs["skip_lines"] = 0  # Start from beginning
        
        # For eval dataset: read from train_split_percentage to end
        eval_kwargs["max_lines"] = eval_lines
        eval_kwargs["skip_lines"] = train_lines  # Skip training data
        
        train_dataset = dataset_class(hstu_config=hstu_config, is_inference=False, **train_kwargs)
        eval_dataset = dataset_class(hstu_config=hstu_config, is_inference=False, **eval_kwargs)
        
        # For iterable datasets, we don't use Subset or DistributedSampler
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffling is handled in the dataset itself
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            # No sampler for IterableDataset
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            # No sampler for IterableDataset
        )
        
        return train_dataloader, eval_dataloader
    
    else:
        # Original logic for map-style datasets
        # Create dataset
        dataset = HammerToTorchDataset(
            dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
        )
        total_items = dataset.dataset.get_item_count() # num of lines in the sequence csv file

        train_size = round(train_split_percentage * total_items)

        train_set = torch.utils.data.Subset(dataset, range(train_size))
        test_set = torch.utils.data.Subset(dataset, range(train_size, total_items))

        # Wrap dataset with dataloader
        train_dataloader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            # batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sampler=DistributedSampler(train_set),
        )
        test_dataloader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            # batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sampler=DistributedSampler(test_set),
        )
        return train_dataloader, test_dataloader


def debug_model_parameters(model: torch.nn.Module, rank: int = 0) -> None:
    """Debug model parameter status"""
    if rank != 0:
        return
    
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    logger.info("=== Model Parameter Debug ===")
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.info(f"TRAINABLE: {name} - shape: {param.shape}")
        else:
            frozen_params += param.numel()
            logger.warning(f"FROZEN: {name} - shape: {param.shape}")
    
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    logger.info("=============================")


def debug_precomputed_module_initialization(model: torch.nn.Module, rank: int = 0) -> None:
    """Debug precomputed module initialization"""
    if rank != 0:
        return
    
    logger.info("=== Precomputed Module Initialization Debug ===")
    
    if hasattr(model, '_precomputed_modules'):
        for name, module in model._precomputed_modules.items():
            logger.info(f"Precomputed module: {name}")
            
            if hasattr(module, 'proj'):
                proj = module.proj
                logger.info(f"  Projection type: {type(proj)}")
                
                # Check if it's a Sequential or Linear
                if isinstance(proj, torch.nn.Sequential):
                    for i, layer in enumerate(proj):
                        if isinstance(layer, torch.nn.Linear):
                            weight = layer.weight
                            bias = layer.bias
                            logger.info(f"    Layer {i} (Linear): weight_shape={weight.shape}")
                            logger.info(f"      Weight stats: mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
                            logger.info(f"      Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
                            if bias is not None:
                                logger.info(f"      Bias stats: mean={bias.mean().item():.6f}, std={bias.std().item():.6f}")
                            logger.info(f"      Requires grad: weight={weight.requires_grad}, bias={bias.requires_grad if bias is not None else 'None'}")
                elif isinstance(proj, torch.nn.Linear):
                    weight = proj.weight
                    bias = proj.bias
                    logger.info(f"  Weight stats: mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
                    logger.info(f"  Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
                    if bias is not None:
                        logger.info(f"  Bias stats: mean={bias.mean().item():.6f}, std={bias.std().item():.6f}")
                    logger.info(f"  Requires grad: weight={weight.requires_grad}, bias={bias.requires_grad if bias is not None else 'None'}")
            
            # Check if module is properly registered
            logger.info(f"  Module device: {next(module.parameters()).device if list(module.parameters()) else 'No parameters'}")
            logger.info(f"  Module training mode: {module.training}")
    
    logger.info("=============================================")

@gin.configurable
def train_loop(
    rank: int,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: Optimizer,
    train_metric_logger: MetricsLogger,
    eval_metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    dataset: str,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 10,
    save_best_model: bool = False,
    print_frequency: int = 100,
    tb_log_dir: str = None,
    # enable_health_monitoring: bool = False,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None, 
    eval_frequency: int = 1000, 
    # lr_scheduler: to-do: Add a scheduler
) -> None:

    # if exp_name:
    #     train_metric_logger.exp_name = exp_name

    # if eval_dataloader is not None and exp_name:
    #     eval_metric_logger.exp_name = exp_name.replace("train", "eval", 1)


    # # Debug model parameters
    # if rank == 0 and enable_health_monitoring:
    #     debug_model_parameters(model, rank)
    #     debug_precomputed_module_initialization(model, rank)

    model = model.train()
    batch_idx: int = 0
    profiler = Profiler(rank, active=10) if output_trace else None

    best_auc = float("-inf")
    best_batch_idx = 0

    print_interval_start_time = time.time()

    # # Log initial model state
    # if rank == 0 and enable_health_monitoring:
    #     total_params = sum(p.numel() for p in model.parameters())
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     logger.info(f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    for epoch in range(num_epochs):

        epoch_losses = []

        for sample in train_dataloader:
            # batch_start_time = time.time()

            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            # pyre-ignore
            loss = sum(aux_losses.values())
            epoch_losses.append(loss.item())
    

            batch_loss = loss.item()
            # writer.add_scalar("Loss/batch", batch_loss, batch_idx)

            if batch_idx > 0 and batch_idx % print_frequency == 0 and rank == 0:
                # Print loss and time
                interval_time = time.time() - print_interval_start_time
                
                print(f"------- Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}] -------")
                print(f"------- Loss: {batch_loss:.6f} | {print_frequency} batches time: {interval_time:.4f}s -------")

            
                # Reset the print interval timer
                print_interval_start_time = time.time()

            loss.backward()


            # # Add training health monitoring AFTER backward pass
            # if enable_health_monitoring:
            #     try:
            #         track_training_health(model, optimizer, loss, batch_idx, rank)
            #         if batch_idx % 100 == 0:
            #             test_communication_in_training(rank, device)
            #     except Exception as e:
            #         logger.warning(f"Health monitoring failed at step {batch_idx}: {e}")
            

            optimizer.step()
            optimizer.zero_grad()

            train_metric_logger.update(
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
            )
            if batch_idx % metric_log_frequency == 0:
                train_metric_logger.compute_and_log(
                    additional_logs={
                        "losses": aux_losses,
                    }
                )
            

            if (eval_dataloader is not None and 
                batch_idx > 0 and 
                batch_idx % eval_frequency == 0):
                if rank == 0:
                    print(f"-------- Start evaluation at batch {batch_idx} --------")
                
                # if 'sample' in locals():
                #     del sample
                # temp_vars_to_clear = ['loss', 'aux_losses', 'batch_loss']
                # for var_name in temp_vars_to_clear:
                #     if var_name in locals():
                #         del locals()[var_name]
                torch.cuda.empty_cache()

                eval_metric_logger.reset()
                eval_auc = eval_loop(
                    rank=rank,
                    model=model,
                    dataloader=eval_dataloader,
                    metric_logger=eval_metric_logger,
                    device=device,
                    train_batch_idx=batch_idx,
                )
                if save_best_model and eval_auc > best_auc:
                    best_auc = eval_auc
                    best_batch_idx = batch_idx
                    if rank == 0:
                        print(f"New best model at batch {best_batch_idx} with AUC {best_auc}")
                    save_dmp_checkpoint(model, optimizer, rank)
                
                torch.cuda.empty_cache()
                if rank == 0:
                    print("-------------------------------------------------")
                model.train()


            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
        
        # calculate epoch loss
        # avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        # writer.add_scalar("Loss/epoch_avg", avg_epoch_loss, epoch)
        # if save_best_model and avg_epoch_loss < best_loss:
        #     best_loss = avg_epoch_loss
        #     best_epoch = epoch + 1
        #     if rank == 0:
        #         print(f"New best model at epoch {best_epoch} with loss {best_loss}")
        #     save_dmp_checkpoint(model, optimizer, rank)


        if num_batches is not None and batch_idx >= num_batches:
            break

    # save_dmp_checkpoint(model, optimizer, rank, suffix="_final")
    # writer.close()


@gin.configurable
def eval_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    exp_name: str = None,  # Experiment name for MetricsLogger
    train_batch_idx: Optional[int] = None, # Training batch index for step alignment
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    
    if exp_name:
        metric_logger.exp_name = exp_name

    model = model.eval()
    batch_idx: int = 0
    profiler = Profiler(rank, active=10) if output_trace else None
    metric_logger.reset()

    if train_batch_idx is not None:
        metric_logger.global_step = train_batch_idx

    eval_losses = []

    with torch.no_grad():
        for sample in dataloader:
            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )

            
            loss = sum(aux_losses.values())


            
            # Handle both tensor and scalar cases
            if isinstance(loss, torch.Tensor):
                eval_losses.append(loss.item())
            else:
                eval_losses.append(float(loss))
            
            metric_logger.update(
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
            )
            
            # metric_logger.compute_and_log()

            del mt_target_preds, mt_target_labels, mt_target_weights
            del sample
            
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break

    # Reset global_step to correct train_batch_idx before logging
    if train_batch_idx is not None:
        metric_logger.global_step = train_batch_idx

    avg_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else float('inf')


    all_metrics = metric_logger.compute()
    # Extract first AUC value
    auc_value = None
    for metric_name, metric_value in all_metrics.items():
        if "auc" in metric_name.lower():  # Find AUC metric
            auc_value = metric_value
            break
    if auc_value is None:
        logger.warning("No AUC metric found, using loss instead")
        return -avg_eval_loss # return negative loss, since higher is better
    

    final_aux_losses = {"total_loss": torch.tensor(avg_eval_loss, device=device)}
    metric_logger.compute_and_log(
        additional_logs={
            "losses": final_aux_losses,
        }
    )
    for k, v in metric_logger.compute().items():
        print(f"{k}: {v}")

    
    return auc_value



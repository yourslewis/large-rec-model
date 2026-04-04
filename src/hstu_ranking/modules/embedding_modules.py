import os
import numpy as np
import torch
import abc
from ops.layer_norm import SwishLayerNorm

class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass

# class PrecomputedEmbeddingModule(torch.nn.Module):
class PrecomputedEmbeddingModule(EmbeddingModule):
    def __init__(
        self, 
        embedding_file_path: str,  # Single .npy file path
        output_dim: int,  # hstu_embedding_table_dim, will be defined in dlrm_hstu.py
        preload: bool = False
    ) -> None:
        super().__init__()
        self.embedding_file_path = embedding_file_path
        
        # Load entire file using memory mapping
        self.embedding_array = np.load(embedding_file_path, mmap_mode="r")

        # Auto-detect input_dim from npy file shape
        self.input_dim = self.embedding_array.shape[1]
        self.output_dim = output_dim
        hidden_dim = self.input_dim * 4

        # Get rank information
        rank = -1
        world_size = -1
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
        except:
            pass

        
        # Projection layer: from precomputed dim to target dim
        # self.proj = torch.nn.Linear(self.input_dim, output_dim, dtype=torch.float32)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim, dtype=torch.float32),
            SwishLayerNorm(hidden_dim), 
            torch.nn.Linear(hidden_dim, self.output_dim, dtype=torch.float32)
        )


        # Initialize projection weights
        self._initialize_projection_weights(rank=rank)
        
    def _initialize_projection_weights(self, rank=-1):
        """Initialize projection layer weights properly"""
        for module in self.proj.modules():
            if isinstance(module, torch.nn.Linear):
                # Use Xavier initialization for weights and zero for biases
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_ids: Tensor of shape [K], item IDs (starting from 1)
        Returns:
            Tensor of shape [K, output_dim]
        """

        device = item_ids.device
        
        # Convert to numpy indices (subtract 1 for 0-based indexing)
        indices = (item_ids - 1).cpu().numpy()

        
        # Index the memory-mapped array
        emb_numpy = self.embedding_array[indices]
        
        # Convert to PyTorch tensor and move to device
        emb_cpu = torch.from_numpy(emb_numpy).float()
        emb = emb_cpu.to(device)

        # Ensure projection layer is on the same device as embeddings
        if not emb.device == next(self.proj.parameters()).device:
            self.proj = self.proj.to(emb.device)

        # return emb
        return self.proj(emb)

    


    def debug_str(self) -> str:
        return f"precomputed_emb_file_{os.path.basename(self.embedding_file_path)}"
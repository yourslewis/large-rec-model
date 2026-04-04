import torch
from typing import Dict, Optional
import os
import logging
from modeling.sequential.embedding_modules import (
    MultiDomainPrecomputedEmbeddingModule,
)
from data.reco_dataset import RecoDataset
import time

class CollateFn:
    def __init__(
        self,
        device: torch.device,
        precomputed_embeddings_domain_to_dir = None,
        item_embedding_dim = None,
        dataset: RecoDataset = None,
    ):
        
        self.device = device
        self.embedding_module = None
        
        self.precomputed_embeddings_domain_to_dir = precomputed_embeddings_domain_to_dir
        self.item_embedding_dim = item_embedding_dim

        self.domain_to_item_id_range = dataset.domain_to_item_id_range
        self.domain_offset = dataset.domain_offset
        self.embd_dim = dataset.embd_dim
        self.shard_size = dataset.shard_size

    # TODO: move hyperparameters to config
    def _init_embedding_module(self):
        if self.embedding_module is None:
            logging.info(f"[PID {os.getpid()}] Initializing embedding module")
            self.embedding_module = MultiDomainPrecomputedEmbeddingModule(
                domain_to_item_id_range=self.domain_to_item_id_range,
                shard_dirs=self.precomputed_embeddings_domain_to_dir,  
                preload=False,
                input_dim=self.embd_dim,
                output_dim=self.item_embedding_dim,          
                shard_size=self.shard_size,
                domain_offset=self.domain_offset,
            )

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        self._init_embedding_module()  # safely lazy-init per process

        input_ids_tensor = torch.stack([sample["input_ids"] for sample in batch])
        timestamps_tensor = torch.stack([sample["timestamps"] for sample in batch])
        length_tensor = torch.tensor([sample["length"] for sample in batch], dtype=torch.long)
        type_ids_tensor = torch.stack([sample["type_ids"] for sample in batch])
        user_id_tensor = [sample["user_id"] for sample in batch]

        result = {
            "input_ids": input_ids_tensor,
            "timestamps": timestamps_tensor,
            "lengths": length_tensor,
            "type_ids": type_ids_tensor,
            "user_id": user_id_tensor,
        }

        
        start = time.time()
        # print("embd lookup in collate_fn for batch")
        raw_input_embeddings = self.embedding_module.get_raw_item_embeddings(input_ids_tensor)  # [B, N, D]
        # print(f"Lookup time: {time.time() - start:.4f}s")
        result["raw_input_embeddings"] = raw_input_embeddings
        return result
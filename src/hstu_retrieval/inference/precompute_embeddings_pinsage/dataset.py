"""
Dataset module for PinSage embedding precomputation with memory-efficient shard-by-shard processing.

This module provides classes for loading and processing domain data (ads, items, etc.) from Parquet files
in a distributed manner across multiple ranks. To avoid memory issues with large datasets, it processes
data one shard at a time rather than loading all assigned shards into memory at once.

Key Components:
- ShardIterator: Iterates through assigned shards one at a time, yielding shard metadata
- DomainDataset: PyTorch Dataset that loads a single shard's data for processing
- CollateFn: Collates batches by tokenizing prompts and preparing them for the model

The workflow is:
1. ShardIterator determines which shards this rank should process
2. For each shard, create a DomainDataset that loads only that shard's data
3. Process the shard through the model
4. Write embeddings to disk
5. Move to the next shard (previous shard's data is garbage collected)

This approach keeps memory usage constant regardless of the total number of shards assigned to a rank.
"""

from torch.utils.data import Dataset
import logging
import pandas as pd
import pyarrow.dataset as ds
import math
import torch
import os
import sys
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from inference.util.normalize import normalize_title, normalize_url

PINSAGE_MAX_LEN = 32   # PinSage model was trained with max_length to be 32, with format as "title [SEP] url"


class ShardIterator:
    """
    Iterator that determines which shards should be processed by a given rank and yields them one at a time.
    
    This class handles the distribution of shards across multiple ranks in a distributed setting,
    ensuring each rank processes a disjoint set of shards. It scans the Parquet file to determine
    the index range and calculates which shards belong to which rank.
    
    Args:
        filesystem: The filesystem object (e.g., fsspec filesystem or AzureMachineLearningFileSystem)
        parquet_dir: Directory containing the Parquet file(s)
        rank: The rank of the current process
        world_size: Total number of processes
        shard_size: Number of rows per shard (default: 25 million)
    
    Yields:
        Tuple[int, int, int]: (shard_id, start_index, end_index) for each shard assigned to this rank
    """
    def __init__(
        self,
        filesystem,
        parquet_dir: str,
        rank: int = 0,
        world_size: int = 1,
        shard_size: int = 25_000_000
    ):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.rank = rank
        self.world_size = world_size
        self.shard_size = shard_size
        
        # Find all Parquet files in the directory
        files = self.fs.glob(f"{parquet_dir}/*.parquet")
        if not files:
            raise ValueError(f"No Parquet files found in {parquet_dir}")
        
        logging.info(f"[ShardIterator] Rank {rank}: Found {len(files)} Parquet file(s) in {parquet_dir}")
        
        # Create a PyArrow dataset from all parquet files
        # PyArrow handles multiple files seamlessly as a single logical dataset
        self.dataset = ds.dataset(parquet_dir, format="parquet", filesystem=self.fs)

        logging.info(f"[ShardIterator] Rank {rank}: Scanning index range from dataset")

        # Read only index column to get full index range
        # PyArrow efficiently reads from all files
        index_table = self.dataset.to_table(columns=["index"])
        df_index = index_table.to_pandas()

        # Handle case where dataset contains no rows to avoid NaN min/max
        if df_index.empty:
            raise ValueError(f"No data found in {self.parquet_dir}")
        min_index = df_index["index"].min()
        max_index = df_index["index"].max()
        total_ids = max_index + 1  # because shards start from 0

        # Total number of shards based on absolute index range
        num_shards = math.ceil(total_ids / shard_size)

        # Shards per rank
        shards_per_rank = math.ceil(num_shards / world_size)
        shard_start = rank * shards_per_rank
        shard_end = min((rank + 1) * shards_per_rank, num_shards)

        # Global-aligned shard ranges with shard IDs
        self.shard_info = [
            (i, i * shard_size, min((i + 1) * shard_size, max_index + 1))
            for i in range(shard_start, shard_end)
            if (i + 1) * shard_size > min_index  # skip shards before actual data
        ]

        logging.info(
            f"[ShardIterator] Rank {rank}: Total shards={num_shards}, "
            f"shards per rank={shards_per_rank}, assigned shards={len(self.shard_info)} "
            f"(IDs {shard_start} to {shard_end - 1})"
        )

    def __iter__(self):
        """Iterate through all shards assigned to this rank."""
        for shard_id, start_idx, end_idx in self.shard_info:
            yield shard_id, start_idx, end_idx

    def __len__(self):
        """Return the number of shards assigned to this rank."""
        return len(self.shard_info)


class DomainDataset(Dataset):
    """
    PyTorch Dataset for loading a single shard of domain data from Parquet file(s).
    
    This dataset is designed to be used with a ShardIterator - it loads only one shard's
    worth of data at a time to minimize memory usage. Each row contains a title and URL
    which are normalized and formatted as a prompt for the PinSage model.
    
    Works seamlessly with multiple parquet files - PyArrow's dataset API handles reading
    from multiple files and filtering by index range efficiently.
    
    Args:
        filesystem: The filesystem object to read from
        parquet_dir: Directory containing the Parquet file(s)
        shard_id: ID of the shard to load
        start_idx: Starting index (inclusive) of the shard
        end_idx: Ending index (exclusive) of the shard
        rank: The rank of the current process (for logging)
    """
    def __init__(
        self,
        filesystem,
        parquet_dir: str,
        shard_id: int,
        start_idx: int,
        end_idx: int,
        rank: int = 0,
    ):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.shard_id = shard_id
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.rank = rank

        logging.info(f"[DomainDataset] Rank {rank}: Loading shard {shard_id} (indices {start_idx}–{end_idx - 1})")

        # Load only the rows for this shard from all parquet files
        # PyArrow efficiently reads across multiple files and filters by index
        dataset = ds.dataset(parquet_dir, format="parquet", filesystem=self.fs)
        table = dataset.to_table(
            columns=["index", "title", "url"],
            filter=(ds.field("index") >= start_idx) & (ds.field("index") < end_idx),
        )
        
        self.df = table.to_pandas()
        logging.info(f"[DomainDataset] Rank {rank}: Loaded {len(self.df)} rows for shard {shard_id}")

    def __len__(self) -> int:
        """Return the number of rows in this shard."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Local index within this shard's dataframe
            
        Returns:
            Dictionary containing:
                - prompt: Normalized text in format "title [SEP] url"
                - index: Global index from the original data
        """
        row = self.df.iloc[idx]
        prompt = f"{normalize_title(row['title'])} [SEP] {normalize_url(row['url'])}"
        return {
            "prompt": prompt,
            "index": row["index"],
        }


class CollateFn:
    """
    Collate function for batching samples in a DataLoader.
    
    This function takes a batch of individual samples (each containing a prompt and index)
    and processes them into a format suitable for the PinSage model. It tokenizes the prompts
    with padding and truncation to ensure uniform input size.
    
    Args:
        tokenizer: The tokenizer from the PinSage model
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dictionaries, each containing 'prompt' and 'index'
            
        Returns:
            Dictionary containing:
                - inputs: Tokenized prompts (input_ids, attention_mask)
                - index: Tensor of global indices
        """
        prompts = [item["prompt"] for item in batch]
        indices = [int(item["index"]) for item in batch]

        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=PINSAGE_MAX_LEN,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        return {
            "inputs": encoded,  # for model
            "index": torch.tensor(indices, dtype=torch.long)  # for downstream use
        }


def get_shard_iterator(
    mode: str = "job",
    path: str = "",
    rank: int = 0,
    world_size: int = 1,
    shard_size: int = 25_000_000,
):
    """
    Get a ShardIterator for processing shards one at a time.
    
    This function sets up the filesystem and creates a ShardIterator that will
    yield shard information for the given rank. This is the entry point for
    the memory-efficient shard-by-shard processing.

    Args:
        mode: "local" for local development or "job" for cluster jobs
        path: Path to the data (used in job mode)
        rank: The rank of the current process
        world_size: Total number of processes
        shard_size: Number of rows per shard (default: 25 million)

    Returns:
        ShardIterator: Iterator that yields (shard_id, start_idx, end_idx) tuples
    """
    if mode == 'local':
        # access data during interactive local development
        from azureml.fsspec import AzureMachineLearningFileSystem
        subscription = 'f920ee3b-6bdc-48c6-a487-9e0397b69322'
        resource_group = 'msan-aml'
        workspace = 'msan-retrieval-ranking-aml'
        datastore_name = 'bingads_algo_prod_networkprotection_c08'
        uri = f'azureml://subscriptions/{subscription}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/{datastore_name}'
        fs = AzureMachineLearningFileSystem(uri)
        prefix = 'shares/bingads.hm/local/NativeAds/Relevance/Data/sequential/hstu/v2/ads_vocab_07012025'
    else:
        # access data in jobs from mounted local file system
        import fsspec
        fs = fsspec.filesystem('file')
        prefix = path
    
    return ShardIterator(fs, prefix, rank, world_size, shard_size)


def create_dataset_for_shard(
    filesystem,
    parquet_dir: str,
    shard_id: int,
    start_idx: int,
    end_idx: int,
    rank: int = 0,
):
    """
    Create a DomainDataset for a specific shard.
    
    This is a helper function to create a dataset that loads only one shard's data.
    Works with multiple parquet files in the directory.
    
    Args:
        filesystem: The filesystem object
        parquet_dir: Directory containing the Parquet file(s)
        shard_id: ID of the shard
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)
        rank: The rank of the current process
        
    Returns:
        DomainDataset: A dataset containing only the specified shard's data
    """
    return DomainDataset(filesystem, parquet_dir, shard_id, start_idx, end_idx, rank)
    
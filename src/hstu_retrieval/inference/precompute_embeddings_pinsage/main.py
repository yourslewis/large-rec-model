"""
Main entry point for memory-efficient PinSage embedding precomputation using shard-by-shard processing.

This script precomputes embeddings for domain data (e.g., ads) using a PinSage model in a distributed
manner across multiple GPUs/nodes. To avoid memory issues with large datasets, it processes one shard
at a time rather than loading all data into memory at once.

Architecture:
- Uses PyTorch's torchrun for multi-GPU/multi-node distributed execution
- Each rank processes a disjoint subset of shards
- For each shard: load data → compute embeddings → write to disk → free memory
- Outputs embeddings in both NPY (dense arrays) and TSV (sparse) formats

Key Features:
- Memory-efficient: Only one shard in memory at a time per rank
- Distributed: Shards are evenly distributed across all ranks
- Fault-tolerant: Each shard is written independently
- Dual output formats: NPY for fast loading, TSV for inspection/debugging

Usage:
  Local development:
    conda activate hstu
    torchrun --nproc_per_node=1 main.py --mode=local --pinsage_ckpt_path=/path/to/checkpoint
    
  Cluster job:
    torchrun --nproc_per_node=8 main.py \
      --mode=job \
      --data_path=/path/to/data \
      --pinsage_ckpt_path=/path/to/checkpoint \
      --output_path=/path/to/output

For more details, see README.md.
"""

import logging
import os

import pandas as pd
from typing import List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages

import torch
import torch.multiprocessing as mp
import absl
from absl import app, flags
import mlflow
from dataset import get_shard_iterator, create_dataset_for_shard, CollateFn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# Set absl logging config
absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity('info')               # Only log info and above
absl.logging.set_stderrthreshold('info')         # Output info-level logs to stderr
absl.logging.get_absl_handler().use_absl_log_file(False)  # Avoid .log file creation

# Clear existing Python logging handlers (in case something pre-configured them)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Attach absl's handler to Python logging
logging.root.addHandler(absl.logging.get_absl_handler())
logging.root.setLevel(logging.INFO)

# Suppress logs from azure libraries
logging.getLogger('azure').setLevel(logging.ERROR)


def delete_flags(FLAGS, keys_to_delete: List[str]) -> None:  # pyre-ignore [2]
    keys = [key for key in FLAGS._flags()]
    for key in keys:
        if key in keys_to_delete:
            delattr(FLAGS, key)


delete_flags(flags.FLAGS, ["data_path", "pinsage_ckpt_path", "output_path", "mode"])
flags.DEFINE_string("data_path", None, "Path to the train/eval data, this is only used for job mode")
flags.DEFINE_string("pinsage_ckpt_path", None, "Path to PinSage model checkpoint")
flags.DEFINE_string("output_path", None, "Path to write the artifacts, this is only used for job mode")
flags.DEFINE_string("mode", "job", "local or job.")


FLAGS = flags.FLAGS  # pyre-ignore [5]


def precompute_embeddings_for_shard(
    dataset,
    shard_id: int,
    model,
    tokenizer,
    batch_size: int,
    shard_size: int,
    output_dir: str,
    local_rank: int,
    rank: int,
) -> None:
    """
    Precompute embeddings for a single shard and write results to disk.
    
    This function processes one shard at a time to minimize memory usage. It:
    1. Creates a DataLoader for the shard
    2. Runs batches through the PinSage model to generate embeddings
    3. Accumulates embeddings in a dense array (size = shard_size)
    4. Writes outputs in both NPY and TSV formats
    5. Returns, allowing the shard data to be garbage collected
    
    Args:
        dataset: DomainDataset containing the shard's data
        shard_id: ID of the shard being processed
        model: PinSage model for encoding
        tokenizer: Tokenizer for the model
        batch_size: Number of samples per batch
        shard_size: Total size of each shard (for creating dense array)
        output_dir: Directory to write output files
        local_rank: Local rank (GPU ID) for this process
        rank: Global rank across all processes
    """
    outpath = f"{output_dir}"
    os.makedirs(outpath, exist_ok=True)

    collate_fn = CollateFn(tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    device = local_rank
    model = model.to(device)
    model.eval()

    # Create dense array for this shard (indices [0, shard_size) map to global indices [shard_id*shard_size, (shard_id+1)*shard_size))
    shard_array = np.zeros((shard_size, 64), dtype=np.float16)
    global_indices = []
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(
            dataloader, 
            desc=f"Rank {rank} Shard {shard_id}", 
            position=rank, 
            leave=True,
            disable=False
        ):
            tokens, indices = batch["inputs"].to(device), batch['index']
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            outputs = model(input_ids, attention_mask)
            batch_emb = outputs.cpu().to(torch.float16).numpy()  # [B, D]

            for emb, global_idx in zip(batch_emb, indices):
                global_idx = int(global_idx)
                local_idx = global_idx % shard_size
                shard_array[local_idx] = emb
                global_indices.append(global_idx)
                embeddings.append(emb)

    # ---- Save NPY (dense format) ----
    npy_dir = f"{outpath}/npy"
    os.makedirs(npy_dir, exist_ok=True)
    out_npy = f"{npy_dir}/shard_{shard_id}.npy"
    np.save(out_npy, shard_array)

    # ---- Save TSV (sparse format) ----
    ids_str = pd.Series(global_indices, dtype="string")
    emb_str = [' '.join(format(x, '.6g') for x in row) for row in embeddings]
    df = pd.DataFrame({
        "id": ids_str,
        "embedding": emb_str,
    })
    tsv_dir = f"{outpath}/tsv"
    os.makedirs(tsv_dir, exist_ok=True)
    out_tsv = f"{tsv_dir}/shard_{shard_id}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False, header=False)

    logging.info(
        f"[Rank {rank}] Saved shard {shard_id}: "
        f"{len(embeddings)} embeddings -> {out_npy}, {out_tsv}"
    )


def precompute_embeddings(
    shard_iterator,
    model,
    tokenizer,
    batch_size: int = 512,
    shard_size: int = 25_000_000,
    output_dir: str = ".",
    local_rank: int = 0,
    rank: int = 0,
) -> None:
    """
    Precompute embeddings for all shards assigned to this rank, processing one shard at a time.
    
    This is the main processing loop that iterates through all shards assigned to this rank.
    For each shard, it:
    1. Creates a dataset containing only that shard's data
    2. Processes the shard through precompute_embeddings_for_shard()
    3. Frees memory (the dataset goes out of scope and is garbage collected)
    4. Moves to the next shard
    
    This approach ensures memory usage remains constant regardless of how many shards
    are assigned to each rank.
    
    Args:
        shard_iterator: ShardIterator that yields (shard_id, start_idx, end_idx) tuples
        model: PinSage model for encoding
        tokenizer: Tokenizer for the model
        batch_size: Number of samples per batch
        shard_size: Number of rows per shard
        output_dir: Directory to write output files
        local_rank: Local rank (GPU ID) for this process
        rank: Global rank across all processes
    """
    logging.info(f"[Rank {rank}] Starting to process {len(shard_iterator)} shards")
    
    for shard_id, start_idx, end_idx in shard_iterator:
        logging.info(f"[Rank {rank}] Processing shard {shard_id} (indices {start_idx}–{end_idx - 1})")
        
        # Create dataset for this shard only
        dataset = create_dataset_for_shard(
            filesystem=shard_iterator.fs,
            parquet_dir=shard_iterator.parquet_dir,
            shard_id=shard_id,
            start_idx=start_idx,
            end_idx=end_idx,
            rank=rank,
        )
        
        # Process this shard
        precompute_embeddings_for_shard(
            dataset=dataset,
            shard_id=shard_id,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            shard_size=shard_size,
            output_dir=output_dir,
            local_rank=local_rank,
            rank=rank,
        )
        
        # Dataset goes out of scope here and will be garbage collected
        logging.info(f"[Rank {rank}] Completed shard {shard_id}")
    
    logging.info(f"[Rank {rank}] Finished processing all {len(shard_iterator)} shards")



def main(argv) -> None:  # pyre-ignore [2]
    """
    Main function for distributed embedding precomputation.
    
    Sets up the distributed environment, loads the model, creates a shard iterator,
    and processes all assigned shards one at a time. Uses torchrun for process
    management and distribution.
    """
    # torchrun sets these environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    job_name = os.getenv("AZUREML_RUN_ID")

    data_path = FLAGS.data_path
    pinsage_ckpt_path = FLAGS.pinsage_ckpt_path
    output_path = FLAGS.output_path
    if output_path:
        output_path = f"{output_path}" 
    mode = FLAGS.mode
    
    logging.info(f"world size: {world_size}, rank: {rank}, local_rank: {local_rank}")
    logging.info(f"mode: {mode}, data_path: {data_path}, output_path: {output_path}")

    BATCH_SIZE = 1024
    SHARD_SIZE = 25_000_000  # 25 million rows per shard
    
    # Get shard iterator (determines which shards this rank will process)
    shard_iterator = get_shard_iterator(
        mode=mode,
        path=data_path,
        rank=rank,
        world_size=world_size,
        shard_size=SHARD_SIZE,  
    )

    # Load PinSage model
    logging.info(f"[Rank {rank}] Loading PinSage model from {pinsage_ckpt_path}")
    from modeling.sequential.pinsage.model.PinSageEncoder import PinSageEncoder
    model = PinSageEncoder.load(pinsage_ckpt_path)
    tokenizer = model.tokenizer

    with mlflow.start_run():
        # Process all shards assigned to this rank, one at a time
        precompute_embeddings(
            shard_iterator=shard_iterator,
            model=model,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            shard_size=SHARD_SIZE,
            output_dir=output_path,
            local_rank=local_rank,
            rank=rank,
        )
    
    logging.info(f"[Rank {rank}] All shards completed successfully")

if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    app.run(main)
# HSTU Retrieval - Repository Structure

## Overview

This module implements the **HSTU (Hierarchical Sequential Transduction Unit)** retrieval model from [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) (ICML'24).

The pipeline processes sequences of user-item interactions and learns to predict the next item via a sequential encoder with sampled softmax loss and contrastive negative sampling.

## Directory Tree

```
src/hstu_retrieval/
├── main.py                         # Training entry point (torchrun + gin configs)
├── debug_main.py                   # Debug training entry point
├── keepalive.py                    # Keepalive utility
├── pytest.ini                      # Pytest configuration
├── environment.yml                 # Conda environment (Python 3.10, torch 2.2.1, fbgemm-gpu 1.2.0)
├── environment_dev.yml             # Dev conda environment
├── REPO_STRUCTURE.md               # This file
├── IMPROVEMENTS.md                 # Known improvement areas
│
├── configs/                        # Gin configuration files
│   ├── ads/                        # Production ads configs
│   │   ├── training_data_05012025_next_event_prediction.gin
│   │   ├── training_data_05012025_conditional_next_event_prediction.gin
│   │   ├── training_data_05012025_next_positive_event_prediction.gin
│   │   ├── training_data_07082025_finetune.gin
│   │   ├── training_data_07082025_semantic_next_event_prediction.gin
│   │   ├── training_data_11032025_astrov6_pinsage_next_event_prediction.gin
│   │   ├── eval_07082025_pinsage.gin
│   │   └── eval_07082025_roberta.gin
│   ├── ml-1m/                      # MovieLens 1M configs (legacy, uses train_fn)
│   ├── ml-20m/                     # MovieLens 20M configs (legacy)
│   ├── ml-3b/                      # MovieLens 3B configs (legacy)
│   ├── amzn-books/                 # Amazon Books configs (legacy)
│   └── local/                      # Local testing configs
│       └── local_training.gin      # Config for local dummy data training
│
├── data/                           # Data pipeline
│   ├── dataset.py                  # DatasetV2 - CSV-based dataset (for ml-1m/ml-20m)
│   ├── reco_dataset.py             # get_reco_dataset() - Dataset factory (gin-configurable)
│   ├── eval.py                     # Evaluation: NDCG@K, HR@K, MRR, log perplexity
│   ├── preprocessor.py             # DataProcessor for MovieLens preprocessing
│   ├── item_features.py            # ItemFeatures for MovieLens genre/title/year
│   └── ads_datasets/               # Ads-domain dataset implementations
│       ├── __init__.py
│       ├── collate.py              # CollateFn - batch collation + embedding lookup
│       ├── buffered_shuffle.py     # BufferedShuffleDataset for IterableDataset
│       ├── special_tokens.py       # PADDING_TOKEN=0, MASK_TOKEN=1
│       ├── next_event_prediction/
│       │   ├── next_event_prediction.py      # Train/Eval IterableDataset (parquet)
│       │   └── test_next_event_prediction.py
│       ├── conditional_next_event_prediction/
│       │   ├── conditional_next_event_prediction.py
│       │   └── test_conditional_next_event_prediction.py
│       ├── next_positive_event_prediction/
│       │   ├── __init__.py
│       │   ├── next_positive_event_prediction.py
│       │   └── test_next_positive_event_prediction.py
│       └── semantic_next_event_prediction/
│           └── semantic_next_event_prediction.py  # Current primary dataset format
│
├── modeling/                       # Model implementations
│   ├── __init__.py
│   ├── initialization.py           # truncated_normal init
│   ├── similarity_module.py        # SequentialEncoderWithLearnedSimilarityModule
│   ├── similarity_utils.py         # get_similarity_function (DotProduct, etc.)
│   └── sequential/                 # Sequential models
│       ├── __init__.py
│       ├── hstu.py                 # HSTU model (main encoder)
│       ├── sasrec.py               # SASRec baseline
│       ├── embedding_modules.py    # LocalEmbeddingModule, MultiDomainPrecomputedEmbeddingModule
│       ├── encoder_utils.py        # get_sequential_encoder (gin-configurable)
│       ├── input_features_preprocessors.py  # Positional embeddings + dropout
│       ├── output_postprocessors.py         # L2Norm / LayerNorm postprocessing
│       ├── features.py             # SequentialFeatures dataclass
│       ├── utils.py                # get_current_embeddings
│       ├── layer_norm.py           # Custom LayerNorm / SwishLayerNorm
│       ├── nagatives_sampler.py    # InBatch, RotateInDomainGlobal, Hybrid samplers
│       ├── autoregressive_losses.py # BCELoss base class
│       ├── losses/
│       │   └── sampled_softmax.py  # SampledSoftmaxLoss (primary loss)
│       └── pinsage/                # PinSage embedding model
│           ├── get_started_with_pinsage.ipynb
│           └── model/
│               ├── __init__.py
│               └── PinSageEncoder.py
│
├── trainer/                        # Training pipeline
│   ├── train.py                    # Trainer class (DDP, AdamW, eval, checkpointing)
│   ├── data_loader.py              # create_data_loader (gin-configurable)
│   └── util.py                     # make_model + SequentialRetrieval wrapper
│
├── indexing/                       # Top-K retrieval
│   ├── __init__.py
│   ├── candidate_index.py          # CandidateIndex class
│   └── utils.py                    # get_top_k_module factory
│
├── rails/                          # RAILS similarity module
│   ├── indexing/
│   │   ├── candidate_index.py      # TopKModule abstract class
│   │   ├── mips_top_k.py           # MIPSBruteForceTopK, MIPSBruteForceShardedTopK
│   │   └── mol_top_k.py            # MoLBruteForceTopK
│   └── similarities/
│       ├── module.py               # SimilarityModule
│       ├── layers.py
│       ├── dot_product_similarity_fn.py
│       └── mol/                    # MOL similarity
│
├── inference/                      # Inference pipelines
│   ├── eval_ads/                   # Ads evaluation inference
│   ├── eval_user/                  # User embedding inference
│   ├── precompute_embeddings_pinsage/
│   ├── precompute_embeddings_roberta/
│   └── util/
│       └── normalize.py
│
├── scripts/                        # Utility scripts
│   ├── generate_local_data.py      # Generate local test data (parquet + .npy)
│   ├── evaluate_checkpoint.py      # Standalone checkpoint evaluation
│   ├── run_training.sh             # Training launch script
│   └── run_evaluation.sh           # Evaluation launch script
│
├── tests/                          # Test suite
│   ├── test_eval_metrics.py        # Evaluation metric correctness tests
│   └── test_local_data.py          # Local data format validation tests
│
└── notebooks/                      # Training notebooks (Azure ML)
    ├── train_singularity_pme.ipynb
    ├── train_singularity_corp.ipynb
    ├── train_env.ipynb
    └── train_env_pme.ipynb
```

## Key Data Flow

```
                 Parquet Files (.parquet)           .npy Embedding Shards
                        │                                   │
                        ▼                                   │
        ┌──────────────────────────┐                        │
        │ semantic_next_event_     │                        │
        │ prediction.py            │                        │
        │ (IterableDataset)        │                        │
        │ yields: {input_ids,      │                        │
        │  timestamps, length,     │                        │
        │  ratings, user_id}       │                        │
        └──────────┬───────────────┘                        │
                   │                                        │
                   ▼                                        ▼
        ┌──────────────────────────────────────────────────────┐
        │ CollateFn (collate.py)                               │
        │ - Stacks batch tensors                               │
        │ - Calls MultiDomainPrecomputedEmbeddingModule        │
        │   .get_raw_item_embeddings(input_ids) → [B, N, 64]  │
        └──────────┬───────────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ Trainer.train() (train.py)                           │
        │ - Receives: input_ids [B,N], raw_input_embeddings    │
        │   [B,N,64], timestamps [B,N], lengths [B], ratings   │
        │ - Shifts: input = ids[:,:-1], labels = ids[:,1:]     │
        │ - Calls SequentialRetrieval.forward()                │
        └──────────┬───────────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ SequentialRetrieval.forward() (util.py)              │
        │ 1. EmbeddingModule.forward(raw_embeddings)           │
        │    → projects 64-dim → 50-dim via MLP                │
        │ 2. HSTU encoder (hstu.py)                            │
        │    → multi-block attention with relative bias        │
        │    → produces sequence embeddings [B, N, D]          │
        │ 3. SampledSoftmaxLoss                                │
        │    → samples negatives (InBatch or RotateGlobal)     │
        │    → computes contrastive loss                       │
        └──────────────────────────────────────────────────────┘
```

## Configuration System

The project uses [gin-config](https://github.com/google/gin-config) for hyperparameter management. All key functions are `@gin.configurable`:

| Function | Configurable Parameters |
|----------|------------------------|
| `get_reco_dataset` | `dataset_name`, `experiment_name`, `max_sequence_length`, `positional_sampling_ratio` |
| `make_model` | `main_module`, `embedding_module_type`, `item_embedding_dim`, `sampling_strategy`, `num_negatives`, `temperature`, ... |
| `Trainer` | `local_batch_size`, `eval_batch_size`, `num_epochs`, `learning_rate`, `weight_decay`, `top_k_method`, ... |
| `hstu_encoder` | `num_blocks`, `num_heads`, `dqk`, `dv`, `linear_dropout_rate`, `attn_dropout_rate`, `normalization`, ... |
| `create_data_loader` | `shuffle`, `num_workers`, `prefetch_factor` |

## Evaluation Metrics

Computed by `data/eval.py`:

| Metric | Formula | Description |
|--------|---------|-------------|
| `ndcg_K` | `1/log2(rank+1)` if rank <= K, else 0 | Normalized Discounted Cumulative Gain |
| `hr_K` | `1` if rank <= K, else `0` | Hit Rate |
| `mrr` | `1/rank` | Mean Reciprocal Rank |
| `log_pplx` | Cross-entropy loss | Log Perplexity (training eval only) |

Evaluation samples 10,000 negatives via `RotateInDomainGlobalNegativesSampler` and ranks the positive item against this candidate pool.

## Dataset Types

| Dataset Name | Format | Code Path | Status |
|-------------|--------|-----------|--------|
| `training_data_05012025` | Parquet (Azure) | `next_event_prediction`, `conditional_next_event_prediction`, `next_positive_event_prediction` | Production |
| `training_data_07082025` | Parquet (Azure) | `semantic_next_event_prediction` | Production |
| `training_data_11032025` | Parquet (Azure) | `semantic_next_event_prediction` | Production (latest) |
| `local_data` | Parquet (local) | `semantic_next_event_prediction` | Local testing |
| `ml-1m` | CSV | `DatasetV2` | Legacy (uses old `train_fn`) |
| `ml-20m` | CSV | `DatasetV2` | Legacy |
| `ml-3b` | CSV | `MultiFileDatasetV2` | Legacy |
| `amzn-books` | CSV | `DatasetV2` | Legacy |

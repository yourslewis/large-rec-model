# HSTU Retrieval - Improvement Areas

This document tracks known issues, hardcoded values, and configurable parameters that should be exposed or improved.

## 1. Hardcoded Values That Should Be Configurable

### CollateFn (data/ads_datasets/collate.py)

The `CollateFn._init_embedding_module()` hardcodes several parameters that should come from the dataset config or gin:

| Parameter | Hardcoded Value | Location | Fix |
|-----------|----------------|----------|-----|
| `input_dim` | `64` | Line 34 | Should come from `RecoDataset.embd_dim` |
| `output_dim` | `50` | Line 35 | Should come from `make_model.item_embedding_dim` |
| `shard_size` | `25_000_000` | Line 36 | Should come from `RecoDataset.shard_size` |
| `domain_offset` | `1_000_000_000` | Line 37 | Should come from `RecoDataset.domain_offset` |

**Current workaround**: The CollateFn creates its own `MultiDomainPrecomputedEmbeddingModule` independently of the model's embedding module. These should share configuration.

### RotateInDomainGlobalNegativesSampler (modeling/sequential/nagatives_sampler.py)

The `domain_pools_map` is hardcoded at line 239:

```python
self.domain_pools_map = {0: [(0, 0.5), (3, 0.5)], 1: [(1, 1.0)], 2: [(2, 1.0)]}
```

This mapping determines which embedding pools to sample negatives from for each domain. It should be:
- A gin-configurable parameter on `make_model` or a separate config
- Passed through `SequentialRetrieval.__init__` to the sampler
- Validated against `shard_counts` to ensure all referenced pools exist

### Trainer Eval/Checkpoint Frequency (trainer/train.py)

The evaluation and checkpoint frequency is hardcoded at line 255:

```python
if batch_id % 1000 == 0:
    self._save_snapshot(batch_id)
    self.model.module.negatives_sampler['eval'].rotate()
    self.run_evaluation_with_pplx(batch_id, epoch)
```

The value `1000` should be configurable via gin (e.g., `Trainer.eval_and_save_interval`).

### Supervision Weight Multiplier (trainer/util.py)

Line 390 hardcodes a 32x weight boost for domain 0 (ad) events:

```python
supervision_weights[label_ids < self.domain_offset] *= 32.0
```

This multiplier should be a configurable parameter.

## 2. Parameters Already Gin-Configurable But Not Always Exposed

### hstu_encoder (modeling/sequential/encoder_utils.py)

These parameters exist but are rarely varied in configs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attn_dropout_rate` | `0.0` | Attention dropout (currently always 0) |
| `normalization` | `"rel_bias"` | Options: `"rel_bias"`, `"hstu_rel_bias"`, `"softmax_rel_bias"` |
| `linear_config` | `"uvqk"` | Only `"uvqk"` is implemented |
| `linear_activation` | `"silu"` | Options: `"silu"`, `"none"` |
| `concat_ua` | `False` | Concatenate u and attention output (3x wider output layer) |
| `enable_relative_attention_bias` | `True` | Can disable relative bias |

### Trainer (trainer/train.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_interval` | `1` | Eval every N epochs (unused in current code) |
| `full_eval_every_n` | `1` | Full eval frequency |
| `save_ckpt_every_n` | `10` | Checkpoint save frequency (unused - hardcoded to 1000 batches) |
| `eval_user_max_batch_size` | `None` | Max batch size for user eval |
| `loss_weights` | `{}` | Weights for auxiliary losses |
| `random_seed` | `42` | Random seed |

### make_model (trainer/util.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_activation_checkpoint` | `False` | Gradient checkpointing for loss |
| `l2_norm_eps` | `1e-6` | L2 norm epsilon |
| `pinsage_ckpt_path` | `""` | PinSage checkpoint path |

## 3. Architectural Improvements

### Data Pipeline

- **Key mismatch in next_event_prediction.py**: The dataset returns `"input"` key but `CollateFn` expects `"input_ids"`. The `semantic_next_event_prediction.py` uses the correct `"input_ids"` key. The older dataset classes should be updated.

- **Redundant label computation**: `next_event_prediction.py` computes `label = ad_ids[1:]` but the Trainer recomputes labels by shifting `input_ids`. The dataset's label field is unused by the Trainer.

- **Eval dataset creates labels differently**: In `next_event_prediction.py`, `EvalIterableDataset._process_row` creates `label = ad_ids[-1]` (single item), while `TrainIterableDataset._process_row` creates `label = ad_ids[1:]` (sequence). The Trainer always recomputes labels from `input_ids`, so both are unused.

### Training Loop

- **No gradient clipping**: The training loop doesn't apply gradient clipping, which can help with training stability for large models.

- **No learning rate scheduler**: Only linear warmup is implemented. Consider adding cosine annealing or other schedulers.

- **eval_interval and save_ckpt_every_n are unused**: These Trainer parameters exist but the actual logic uses a hardcoded `batch_id % 1000 == 0` check.

- **Eval batches capped at 100**: `run_evaluation_with_pplx()` breaks after 100 batches (line 360). This should be configurable.

### Negative Sampling

- **RotateInDomainGlobalNegativesSampler.rotate()** loads one shard per domain per rotation. For domains with many shards, this means the eval negative pool only covers a fraction of the item catalog at any time. Consider loading multiple shards or implementing reservoir sampling.

- **10k eval negatives**: `get_eval_state_v2()` samples exactly 10,000 negatives (line 94). This should be configurable.

### Model

- **Autocast dtype is None**: `HSTUJagged` is initialized with `autocast_dtype=None`, meaning no mixed-precision inside the attention layers. The `main_module_bf16` flag converts the whole model to bf16 before sending to GPU, but the HSTU attention could benefit from fine-grained mixed precision control.

- **RelativeBucketedTimeAndPositionBasedBias** uses a fixed bucketization function (`log(abs(x).clamp(min=1)) / 0.301`). The number of buckets (128) and the bucketization function should be configurable.

## 4. Code Quality

- **Typo**: `nagatives_sampler.py` should be `negatives_sampler.py`

- **Dead code in eval.py**: `eval_metrics_v2_from_tensors` accepts `filter_invalid_ids` and `user_max_batch_size` parameters but never uses them.

- **Import organization**: `reco_dataset.py` has duplicate `from typing import List, Optional, Dict, Tuple` imports (lines 5 and 31).

- **Missing type annotations**: Several functions use `# pyre-unsafe` and have incomplete type annotations.

- **eval() usage in dataset.py**: `DatasetV2.load_item()` uses Python's `eval()` to parse string representations of lists (line 72). This is a security risk if data comes from untrusted sources. Consider using `ast.literal_eval()` or storing data in a safer format.

## 5. Operational Improvements

- **No early stopping**: Training always runs for all `num_epochs`. Consider adding early stopping based on eval metrics.

- **No metric logging to file**: Metrics are logged to TensorBoard and MLflow, but there's no simple CSV/JSON export for offline analysis.

- **Checkpoint management**: Old checkpoints are never cleaned up. Consider keeping only the N most recent checkpoints.

- **No distributed evaluation**: Evaluation runs identically on all ranks, with results averaged via `all_reduce`. For large eval sets, sharded evaluation (like `run_sharded_evaluation`) should be the default.

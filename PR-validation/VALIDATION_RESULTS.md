# PR #2 Validation Results: Config-Driven Registry Refactor

## Summary

The refactored code (branch refactor/config-driven-registry) reproduces and slightly exceeds
the original baseline metrics across both sampling strategies tested.

## Experiments

| Experiment | Sampling Strategy | GPU | Steps | Status |
|---|---|---|---|---|
| **2A** | InBatch | RTX 4090 #1 | 172K | Complete |
| **3D** | RotateInDomain | RTX 4090 #0 | 75K | Complete |

**Server:** 2x RTX 4090 (24GB), 188GB RAM
**Dataset:** lrm_astrov6_split (ads, web, shopping domains)

## Final Results

| Metric | 2A Refactored | 2A Baseline | 3D Refactored | 3D Baseline |
|---|---|---|---|---|
| **NDCG@10** | **0.3599** | 0.3433 | **0.4902** | 0.4776 |
| **HR@10** | **0.5171** | 0.5015 | **0.6252** | -- |
| **MRR** | **0.3224** | 0.3025 | **0.4567** | -- |
| Steps | 172K | -- | 75K | 75K |
| Step time | 1.26s | -- | 0.2s | -- |
| Eval time | 28s | -- | 31s | -- |

Both refactored experiments exceed their respective baselines.

## 3D Training Curve (NDCG@10)

| Step | NDCG@10 | HR@10 | MRR |
|------|---------|-------|------|
| 0 | 0.1539 | 0.2022 | 0.1437 |
| 5K | 0.3984 | 0.5385 | 0.3649 |
| 10K | 0.4166 | 0.5658 | 0.3799 |
| 20K | 0.4540 | 0.5934 | 0.4197 |
| 40K | 0.4695 | 0.6127 | 0.4340 |
| 60K | 0.4779 | 0.6292 | 0.4391 |
| 75K | 0.4902 | 0.6252 | 0.4567 |

## Configuration

Both experiments use SampledSoftmaxLoss with:
- softmax_temperature = 0.05
- num_negatives = 128 (train)
- eval: 10K sampled negatives + MIPSBruteForceTopK
- eval_batch_size = 128, eval_max_batches = 100

### 2A (InBatch)
- Negatives drawn from current batch items
- Config: PR-validation/configs/validation_2A.gin

### 3D (RotateInDomain)
- Negatives from pre-loaded global corpus shards (25M items x 64d x fp16 per shard)
- 3 domains: ads, web, shopping
- Shards rotated every 1K steps via rotate()
- Config: PR-validation/configs/validation_3D.gin

## Bug Fix: Zombie Worker Prevention

Added PR_SET_PDEATHSIG to DataLoader worker_init_fn in data_loader.py.
When the parent training process dies (OOM kill, crash, manual kill), Linux
automatically sends SIGTERM to all DataLoader workers, preventing orphan
processes from accumulating and exhausting system RAM.

# PR #1 Validation Results

**PR:** astrov6-local-changes → upstream-latest
**Date:** 2026-04-05/06
**Hardware:** 2× NVIDIA RTX 4090 (24GB each)
**Data:** astrov6 parquet, 15 train shards (~117GB), 1 eval shard (~8GB held-out shard 15)
**Seed:** 42 (all runs)

---

## Phase 1: Baselines (single-GPU, batch=128)

Both codebases trained on identical data with matched hyperparameters for 75K iterations.

| Run | Description | NDCG@10 | HR@10 | MRR | log_pplx |
|-----|-------------|---------|-------|-----|----------|
| 2A | Upstream baseline (event type emb + MLP proj + InBatch) | 0.3433 | 0.4944 | 0.3077 | 1.6976 |
| 2B | Our code (no event type emb + Linear-SwishLN proj + InBatch) | 0.3317 | 0.4865 | 0.2952 | 1.6555 |

**Gap:** 2A leads by 3.38 % on NDCG@10

---

## Phase 2: Feature Ablations (DDP×2, batch=128/GPU)

All ablations modify the upstream (2A) code to isolate a single feature's impact.
DDP×2 means each iteration processes 2× the samples — 37K DDP iters ≈ 74K single-GPU iters.

| Run | Feature Ablated | NDCG@10 | HR@10 | MRR | log_pplx | Δ NDCG vs 2A |
|-----|----------------|---------|-------|-----|----------|--------------|
| 3A | F2: event type embedding removed | 0.3270 | 0.4776 | 0.2921 | 1.7210 | -4.75% |
| 3B | F1: embedding proj → Linear-SwishLN(1024) | 0.3533 | 0.5057 | 0.3173 | 1.6311 | +2.91% |
| 3D | Neg sampling → RotateInDomain (1280 negs) | 0.4776 | 0.6184 | 0.4426 | 2.2387 | +39.12% |

---

## 3B Extended Training (ran to 107K DDP iters)

| DDP Iter | ~Single-GPU Equiv | NDCG@10 | HR@10 | MRR | log_pplx |
|----------|-------------------|---------|-------|-----|----------|
| 37000 | ~74000 | 0.3533 | 0.5057 | 0.3173 | 1.6311 |
| 50000 | ~100000 | 0.3565 | 0.5093 | 0.3204 | 1.6099 |
| 75000 | ~150000 | 0.3633 | 0.5179 | 0.3265 | 1.5914 |
| 100000 | ~200000 | 0.3606 | 0.5131 | 0.3246 | 1.5787 |
| 107000 | ~214000 | 0.3672 | 0.5197 | 0.3310 | 1.5777 |

---

## Attribution Summary

| Feature | Effect | Verdict |
|---------|--------|---------|
| **F2: Event type embedding** | Removing it drops NDCG by ~4.75% | **Keep it** — meaningful signal |
| **F1: Embedding projection** | Our wider proj (1024) improves NDCG by ~2.91% | **Our version is better** |
| **RotateInDomain negatives** | +39% NDCG — dramatically better retrieval | **Strong improvement** (but higher pplx, different loss landscape) |
| F3: model_hidden_size + output proj | DROPPED — no-op when hidden=item_emb_dim=128 | N/A |

## Key Takeaways

1. **The 2A→2B gap (~3.4%) is primarily explained by removing event type embedding (F2).** Our projection change (F1) actually *helps*, so it partially offsets the F2 loss.
2. **RotateInDomain negative sampling is the single biggest improvement available** — but it changes the loss landscape fundamentally (higher pplx, much better retrieval metrics).
3. **Our wider embedding projection (1024 hidden) outperforms upstream's MLP** at matched sample count.

## Configs Used

| Test | Config File | Code Branch | Key Differences from 2A |
|------|-------------|-------------|------------------------|
| 2A | validation_2A.gin | upstream-latest | Baseline |
| 2B | validation_2B.gin | astrov6-local-changes | No event type emb, different proj, InBatch |
| 3A | validation_3A.gin | upstream-latest (modified) | `LearnablePositionalEmbeddingInputFeaturesPreprocessor` (no event type) |
| 3B | validation_3B.gin | upstream-latest (modified) | Proj swapped to Linear(64→1024)→SwishLN→Linear(1024→128)→LN |
| 3D | validation_3D.gin | upstream-latest (modified) | `RotateInDomainGlobalNegativesSampler`, 1280 negatives |

## Reproduction

```bash
# From repo root:
cd PR-validation
./run_tests.sh setup        # Prepare data split
./run_tests.sh launch 2A    # Run a specific test
./run_tests.sh metrics 2A   # Check results
./run_tests.sh report       # Full comparison
./run_tests.sh status       # GPU status
```

# Training Sequential Models

## Overview
This guide walks you through training a sequential recommendation model in three steps: event ID assignment, embedding precomputation, and model training.

## Quick Reference

| Step | What It Does | Where to Run | When to Re-run |
|------|--------------|--------------|----------------|
| **1. Event ID Assignment** | Converts event sequences to ID sequences | Magnetar (PySpark) | Dataset changes |
| **2. Embedding Precomputation** | Generates embeddings for all events | GPU (inference notebook) | Input embedding model changes |
| **3. Model Training** | Trains user embedding model | GPU (`train_singularity_pme.ipynb`) | Model architecture changes |

## Training Steps

### Step 1: Assign an ID to Each Event

**Purpose:** Convert sequential event data into compact ID sequences for efficient GPU processing.

**Input:**  
Dataset where each row contains a sequence of events:
```python
[(event_1_type, event_1_title, event_1_url), 
 (event_2_type, event_2_title, event_2_url), 
 ...]
```

**Outputs:**
1. **Event Dictionary (E):** Maps unique events to IDs
   ```python
   {
       128: (event_1_type, event_1_title, event_1_url),
       231: (event_2_type, event_2_title, event_2_url),
       ...
   }
   ```

2. **ID-based Training Data:** Sequences of IDs instead of event tuples
   ```python
   [121, 372, ...]
   ```

**Execution:** Run PySpark notebook in Magnetar.

---

### Step 2: Precompute Input Embeddings

**Purpose:** Generate embeddings for all unique events using your chosen embedding model.

**Input:** Event dictionary (E) from Step 1

**Process:** Compute embeddings using models like:
- PinSage
- XLM-RoBERTa
- Gemma

**Output:** Precomputed embeddings for all events

**Execution:** Run GPU notebook in `inference/precomputed_embeddings/` folder.

---

### Step 3: Train the Model

**Purpose:** Train the sequential model to learn user embeddings from event sequences.

**Input:** ID-based training data + precomputed embeddings

**Output:** Trained model weights

**Execution:** Run `train_singularity_pme.ipynb` notebook.

---

## When to Re-run Steps

- **Only Step 3:** Experimenting with model architecture changes (dataset and embeddings unchanged)
- **Steps 2 & 3:** Testing different embedding models (dataset unchanged)
- **All Steps:** Working with a new dataset


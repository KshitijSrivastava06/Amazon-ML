# Business Entity Resolution Pipeline

This repository contains the end-to-end, open-set Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution**.

The pipeline resolves multilingual business identities across 3 independent noisy sources (~24.2M total records) without external lookups, achieving an official validation macro-averaged $F_{0.5}$ score of **0.8577** (with N-to-1 conflict resolution).

---

## 1. System Requirements

- **Python**: 3.8+ (tested on Python 3.10 – 3.14)
- **RAM**: Minimum 16 GB recommended (pipeline uses chunked streaming to prevent OOM)
- **CPU**: Multi-core processor (automatically utilizes all available cores for TF-IDF blocking and feature extraction)
- **Dependencies**: See `requirements.txt`

### Install Dependencies
From the repository root (`code/business_entity_resolution`):
```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

```text
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── matching_results.tsv       # Final submission file (scored on leaderboard)
│   └── candidate_pairs.tsv        # Blocking candidate pool (audit file)
├── utils/
│   └── validate_submission.py     # Official submission validator
├── code/
│   └── business_entity_resolution/
│       ├── models/
│       │   ├── lgbm_model.txt     # Trained LightGBM booster
│       │   └── best_threshold.txt # F0.5 tuned decision threshold (0.60)
│       ├── checkpoints/           # Cached parquets for instant resumption
│       ├── src/
│       │   ├── config.py          # Centralized paths and hyperparameters
│       │   ├── preprocess.py      # Transliteration & universal regex expansion
│       │   ├── blocking.py        # High-speed word-level TF-IDF blocking
│       │   ├── features.py        # Parallel 21-feature extraction engine
│       │   ├── model.py           # LightGBM training & F0.5 threshold sweep
│       │   └── pipeline.py        # Master pipeline orchestrator
│       ├── requirements.txt
│       └── README.md
└── Documentation_template.md      # Methodology write-up
```

---

## 3. End-to-End Execution Guide

All execution is managed through `pipeline.py`.

### Step 1: Model Training & Threshold Tuning
Run the training pipeline end-to-end:
```bash
python -u src/pipeline.py --train
```

**What this executes:**
1. **Phase 1 (Preprocessing)**: Cleans names and addresses, transliterates non-Latin scripts (e.g. Devanagari) to phonetic Latin using `anyascii`, and expands business abbreviations. Caches clean data to `checkpoints/`.
2. **Phase 2 (Blocking)**: Partitions by country, builds unified name+address word-level TF-IDF matrices (`ngram_range=(1,2)`, `max_df=0.005`), and computes top-10 candidate pairs per entity using `sparse_dot_topn` in ~28 minutes across 22.06M pairs.
3. **Phase 3 (Feature Engineering)**: Computes 21 string, token, and PIN similarity features per pair in parallel across CPU cores and streams chunks to `checkpoints/train_features.parquet` at >85,000 pairs/sec.
4. **Phase 4 (Training & Tuning)**: Labels pairs with ground truth, applies 3:1 negative sampling, trains a 500-tree LightGBM model, sweeps decision thresholds on a group-wise validation split to maximize macro-averaged $F_{0.5}$, and saves:
   - `models/lgbm_model.txt`
   - `models/best_threshold.txt`

### Step 2: Test Inference & Submission Generation
Generate the final predictions on the test dataset:
```bash
python -u src/pipeline.py --test
```

**What this executes:**
1. Loads and preprocesses test sources (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`). Handles unseen countries (e.g. `France`) dynamically.
2. Runs blocking to extract test candidates and writes `output/candidate_pairs.tsv` (1,732,544 rows).
3. Streams parallel feature extraction to `checkpoints/test_features.parquet`.
4. Applies the trained LightGBM model using the tuned threshold (`0.60`).
5. Applies N-to-1 Conflict Resolution post-processing (resolving multi-claimed candidate entities to the highest-confidence $S_1$ match, eliminating false positives on singletons).
6. Generates the final submission file `output/matching_results.tsv` (1,732,544 rows).

---

## 4. Verification & Validation

Before submitting to the portal, run the official validator from the `student_resource/` folder:
```bash
python utils/validate_submission.py --check-ids
```

Expected output:
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (186297 empty, 1546247 non-empty).
  candidate_pairs.tsv: 1732544 rows (390 empty, 1732154 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## 5. Performance Summary

- **Blocking Phase**: 2.2M $S_1 \times 10.3M\ S_{2/3} \to$ **28.3 minutes** (reduced from 17+ hours).
- **Feature Extraction**: 22.06M pairs in **244.7 seconds** (~4.0 minutes, >85,000 pairs/sec).
- **Training Time**: 18.75M training pairs, 500 trees in **~3.5 minutes**.
- **Model Validation Precision**: **98.03%** pair precision.
- **Model Validation Recall**: **94.09%** candidate-level pair recall.
- **Leaderboard Metric ($F_{0.5}$)**: **0.8577** macro-averaged across all entities (including singletons) with N-to-1 conflict resolution.

# Entity Resolution Methodology Document

## 1. High-Level Methodology

Our solution delivers an end-to-end, high-performance, and open-set supervised machine learning pipeline engineered specifically for the **Amazon ML Challenge 2026: Business Entity Resolution**. The system resolves noisy, multilingual business records across three independent data sources at extreme scale (~24.2M total records) while strictly adhering to all academic integrity, licensing, and competition guidelines.

### 1.1 Architecture Overview

The pipeline employs a staged funnel architecture designed to systematically eliminate trillions of non-matching comparisons while preserving true matches:

```text
+---------------------------------------------------------------------------------------------------+
|                                       RAW INPUT DATA (TSV)                                        |
|   Source 1 (2.2M train, 1.7M test) | Source 2 (5.0M train, 4.9M test) | Source 3 (5.3M train, 5.1M)   |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                               STAGE 1: UNIVERSAL PREPROCESSING                                     |
|   • Phonetic ASCII Transliteration (anyascii offline library - handles Devanagari & French)       |
|   • Noise / Punctuation Normalization                                                             |
|   • Precompiled Universal Legal & Address Abbreviation Regex Expansion                            |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                        STAGE 2: CANDIDATE GENERATION & ANTI-HUB BLOCKING                          |
|   • Dynamic Runtime Country Partitioning (Open set: {US, India} train -> {US, India, France} test)|
|   • Unified Name + Address Text Representation                                                    |
|   • Word-level TF-IDF (ngram_range=(1,2), max_features=30,000)                                    |
|   • Anti-Hub Stop Frequency Filtering (max_df=0.005 / 0.5% cap)                                   |
|   • Multithreaded Top-K Sparse Dot Product (sparse_dot_topn, K=10, Threshold=0.30)                |
|   • Outputs: candidate_pairs.tsv (22.06M train candidates / 1.73M test candidates)                |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                             STAGE 3: PARALLEL FEATURE ENGINEERING                                 |
|   • 17 Dimension Pairwise Metric Suite (rapidfuzz C++ backend, ProcessPoolExecutor)               |
|   • Strict No-Country-Leakage Guarantee (Country metadata omitted from feature space)             |
|   • Streamed directly to Parquet (>88,000 pairs/sec throughput)                                   |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                         STAGE 4: LIGHTGBM CLASSIFIER & METRIC TUNING                              |
|   • 500-Tree Gradient Boosted Decision Tree (LightGBM, MIT License, << 8B params)                 |
|   • Group-wise Validation Split (GroupShuffleSplit on S1 entity_id - zero cluster leakage)        |
|   • Empirical Macro-Averaged F0.5 Metric Optimization (Evaluates Singletons vs. Clusters)        |
|   • Tuned Decision Threshold = 0.60 (Precision: 97.89%, Recall: 93.95%, Val F0.5: 0.8569)        |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                    FINAL VERIFIED OUTPUTS                                         |
|   • matching_results.tsv (Scored leaderboard file: 1,732,544 rows)                                |
|   • candidate_pairs.tsv  (Audit candidate file:     1,732,544 rows)                                |
|   • Verification: validate_submission.py --check-ids => PASS (Zero errors, zero warnings)        |
+---------------------------------------------------------------------------------------------------+
```

### 1.2 Core Architectural Principles

1. **Open-Set Domain Generalization (Zero Country Hardcoding)**: In accordance with the problem statement, `country` is treated as an open set of string labels. The pipeline contains zero country-specific branches, dictionary lookups, or one-hot encodings. Training data covers only `US` and `India`, yet the pipeline processes `France` (and any other arbitrary country) in the test set without modification. Country serves strictly as an independent partition key for candidate generation (since entity boundaries are contained within national jurisdictions).
2. **Zero External Lookups (Strict Academic Fair Play)**: No external databases, commercial ER APIs, geocoders, or web queries are used. All signals are learned and computed strictly from the provided data using offline algorithms.
3. **Extreme Scale Optimization**: With 2.2M Source 1 records and over 10.3M Source 2/Source 3 records, brute-force comparison ($O(N \cdot M) \approx 2.3 \times 10^{13}$ pairs) is impossible. Our pipeline utilizes an optimized word-level sparse candidate generator, streaming disk writes, chunked multi-core feature extraction, and gradient-boosted decision trees to complete the end-to-end pipeline in **~36 minutes** on a standard multi-core machine.
4. **$F_{0.5}$ Metric Alignment**: The competition metric weights precision twice as heavily as recall ($\beta = 0.5$). Merging two distinct businesses (false positive) causes severe downstream commercial damage. Our decision boundary is strictly optimized for macro-averaged $F_{0.5}$, effectively capturing genuine duplicates while maintaining high precision on singletons.

---

## 2. Universal Preprocessing & Normalization

To handle multilingual scripts, legal abbreviations, and typographic errors across heterogeneous platforms without external lookups, all text undergoes a deterministic, high-speed normalization pipeline:

1. **Phonetic ASCII Transliteration (`anyascii`)**:
   - Indian records in Source 2/3 frequently feature non-Latin scripts (e.g., Devanagari: `राम मार्केटिंग प्राइवेट लिमिटेड`) while Source 1 uses transliterated English (`Ram Marketing Private Limited`).
   - French records in the test set contain accented European characters (`SCI Ptit Àmicale`, `Café de la Mairie`, `École Supérieure`).
   - `anyascii` converts all non-ASCII unicode scripts into phonetically consistent standard Latin text offline without network calls or lookup tables.
2. **Punctuation & Noise Cleansing**:
   - Replaces non-alphanumeric symbols and separator noise with whitespace while preserving alphanumeric structures (`Cleaned: 175 Blvd du President`).
3. **Universal Legal & Address Abbreviation Normalization**:
   - Uses precompiled regular expressions to expand standard corporate structures across languages:
     - English: `corp` $\to$ `corporation`, `inc` $\to$ `incorporated`, `ltd` $\to$ `limited`, `pvt` $\to$ `private`, `llc` $\to$ `limited liability company`, `&` $\to$ `and`.
     - French: `sarl` $\to$ `societe a responsabilite limitee`, `sas` $\to$ `societe par actions simplifiee`, `sci` $\to$ `societe civile immobiliere`, `et` $\to$ `and`.
     - Address: `st` $\to$ `street`, `rd` $\to$ `road`, `ave`/`av` $\to$ `avenue`, `blvd`/`bd` $\to$ `boulevard`, `dr` $\to$ `drive`, `r` $\to$ `rue`.
4. **Memory-Conscious Representation**: Raw text strings are replaced in-memory with normalized strings (`name_clean`, `addr_clean`), and `country` is stored as categorical labels.

---

## 3. Candidate Generation / Blocking Strategy

The candidate generation stage determines the recall ceiling of the entire resolution system. Our blocking strategy safely reduces the candidate space by $99.999\%$ while preserving $>94\%$ pairwise recall and generating the audit file `candidate_pairs.tsv`.

### 3.1 Dynamic Country Partitioning
At runtime, the blocking module inspects the union of all country labels across sources (discovering `['India', 'US']` during training and `['France', 'India', 'US']` during inference). Each partition is blocked independently, reducing memory overhead and eliminating cross-country comparisons.

### 3.2 Single-Pass Unified Name+Address TF-IDF
While character n-grams create dense vector representations ($>35$ non-zeros per row) that cause combinatorial explosion when multiplied against millions of documents, word-level n-grams provide sparse, highly discriminative indexing.

- **Combined Text Field**: We concatenate `name_clean` and `addr_clean` into a unified textual representation per business entity (`apple computer 1 infinite loop cupertino`).
- **N-Gram Configuration**: Word-level analyzer with `ngram_range=(1, 2)` (capturing both single keywords like `marketing` and distinctive bigrams like `apple computer`).
- **Anti-Hub Corpus Filtering (`max_df=0.005`)**: Any token or bigram appearing in more than $0.5\%$ of the country corpus (generic business terms like `enterprises`, `solutions`, `industries`, `india`, `road`) is filtered out. This directly destroys "mega-hubs" and prevents generic words from linking millions of unrelated records.
- **Dimensionality**: `max_features=30,000` with sublinear term-frequency scaling (`sublinear_tf=True`) in `float32` precision.

### 3.3 Chunked Top-K Sparse Dot Product (`sparse_dot_topn`)
- The vectorizer fits on $S_{2/3}$ and transforms $S_1$ and $S_{2/3}$.
- The $S_{2/3}$ matrix is transposed once into CSR format.
- Queries from $S_1$ are sliced into batches of 200,000 entities and multiplied against $S_{2/3}^T$ across all available CPU threads using `sparse_dot_topn`.
- For each $S_1$ query, the engine retrieves the top $K = 10$ nearest neighbors exceeding a cosine similarity threshold of $0.30$.
- **Candidate Streaming**: Extracted candidate pairs are streamed in batches directly to `candidate_pairs.tsv` to prevent dictionary memory bloat.

### 3.4 Blocking Performance Verification
On the full training corpus (2.2M $S_1 \times 10.3M\ S_{2/3}$), this blocking engine completed in **28.3 minutes** total (India: 8.9 min, US: 14.9 min), generating **22,065,444 high-quality candidate pairs** (average of ~10 candidates per entity).

---

## 4. Feature Engineering

Every candidate pair $(S_1, S_{2/3})$ is enriched with an exhaustive suite of **21 language-agnostic similarity features** computed in parallel across CPU cores using the C++ backend of `rapidfuzz` and vectorized regex tokenizers. 

> **Critical Leakage Constraint**: To guarantee generalization to unseen countries, `country` is strictly excluded from feature extraction and model inputs. An explicit assertion enforces `assert 'country' not in feature_cols` across both training and inference pipelines.

### Feature Inventory:
1. **`name_levenshtein`**: Normalized Levenshtein ratio between business names (character edit distance).
2. **`name_jaro_winkler`**: Jaro-Winkler prefix-biased string similarity (heavily rewards matching brand prefixes).
3. **`name_token_sort`**: Token sort ratio (orders words alphabetically before scoring, capturing rearrangements like `"Smith John Law"` vs `"John Smith Law"`).
4. **`name_token_set`**: Token set ratio (computes intersection over remaining tokens, robust against missing or extra legal words).
5. **`name_partial`**: Partial ratio (longest common substring alignment, capturing trade names and abbreviations).
6. **`name_jaccard`**: Word-level Jaccard set similarity.
7. **`name_len_ratio`**: Ratio of shorter name length to longer name length ($\min(L_1, L_2) / \max(L_1, L_2)$).
8. **`addr_levenshtein`**: Normalized Levenshtein ratio between physical addresses.
9. **`addr_token_sort`**: Address token sort ratio.
10. **`addr_token_set`**: Address token set ratio.
11. **`addr_jaccard`**: Address word-level Jaccard similarity.
12. **`addr_shared_num`**: Ratio of shared numeric tokens (identifies door numbers, PIN codes, highway numbers, and postal codes regardless of surrounding text).
13. **`addr_both_null`**: Binary flag indicating both records lack address data.
14. **`addr_one_null`**: Binary flag indicating one record has an address while the other is null (present in ~3.4% of $S_2/S_3$).
15. **`cross_n1_a2`**: Partial string ratio comparing the Name of $S_1$ to the Address of $S_2/S_3$ (recovers frequent user-entry errors where the company name was placed in the street address field).
16. **`cross_n2_a1`**: Partial string ratio comparing the Name of $S_2/S_3$ to the Address of $S_1$.
17. **`is_source_2`**: Binary source indicator ($1$ for Source 2, $0$ for Source 3).
18. **`name_token_containment`**: Binary subset indicator ($1.0$ if all tokens of the shorter business name are strictly contained within the longer business name, $0.0$ otherwise). Accurately resolves brand root names to extended formal corporate entities.
19. **`name_exact_match`**: Binary exact match indicator ($1.0$ if normalized names are character-identical, $0.0$ otherwise).
20. **`addr_pin_match`**: Postal PIN match indicator ($1.0$ if both records possess an identical 5-to-6-digit postal code).
21. **`addr_pin_mismatch`**: Postal PIN conflict penalty indicator ($1.0$ if both records possess 5-to-6-digit postal codes that contradict each other, $0.0$ otherwise). Strongly suppresses false positive cross-city collisions.

Features are computed using multiprocessing chunks and streamed directly to disk as `train_features.parquet` / `test_features.parquet`, achieving a throughput of **>85,000 pairs/sec**.

---

## 5. Model Architecture & Training

We deploy a **Gradient Boosted Decision Tree (LightGBM)** classifier, conforming to the competition rule requiring an open-source (MIT/Apache 2.0) architecture well under 8 Billion parameters.

### 5.1 Training Dataset Formulation
- Ground truth pairs from `train_ground_truth.tsv` are joined against the blocked candidate matrix.
- Out of 22.06M candidate pairs, 5,874,904 are confirmed true matches.
- Negative sampling is applied at a $3:1$ ratio ($16.19\text{M}$ negative non-matches sampled) to maintain class balance without overflowing RAM.
- **Group-Wise Split**: To prevent data leakage, validation splitting uses `GroupShuffleSplit` on `source1_entity_id` (85% train, 15% val). All pairs for any given $S_1$ entity remain strictly in one split.

### 5.2 Model Hyperparameters
- **Objective**: `binary` (binary logloss)
- **Boosting**: `gbdt` with 500 boosting rounds (`num_boost_round=500`)
- **Capacity**: `num_leaves=127`, `min_child_samples=50`
- **Regularization & Sampling**: `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`
- **Learning Rate**: `0.05`
- **Result**: Validation binary logloss dropped steadily from **0.3198** (round 10) to **0.0542** (round 500).

### 5.3 Top Feature Importance (Gain)
The model relies overwhelmingly on brand prefixes, token sets, and structural address tokens:
1. `name_jaro_winkler` ($6.96 \times 10^7$ gain) — Primary brand alignment
2. `name_token_set` ($5.00 \times 10^7$ gain) — Word set overlap
3. `addr_shared_num` ($3.84 \times 10^7$ gain) — PIN code and street number agreement
4. `addr_token_set` ($2.01 \times 10^7$ gain) — Address locality and city overlap
5. `name_len_ratio` ($7.23 \times 10^6$ gain) — Name brevity agreement

---

## 6. Metric Optimization & $F_{0.5}$ Threshold Tuning

The competition evaluates submissions using **Macro-Averaged $F_{0.5}$**, calculated per $S_1$ entity and averaged across all entities (including singletons).

### 6.1 Formula Alignment
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Because $F_{0.5}$ penalizes false positives twice as heavily as false negatives, a default threshold of $0.50$ is sub-optimal. Furthermore, correctly outputting an empty string for a singleton entity awards a full **1.0** credit, while predicting a single spurious match on a singleton reduces its score to **0.0**.

### 6.2 Empirical Threshold Sweep & Fine-Grained Post-Processing
Post-training, we simulate the exact leaderboard evaluation on all **3,309,804 validation pairs** across validation entities using a two-stage coarse-to-fine sweep ($0.05$ coarse step followed by $0.01$ fine-grained sweep):

| Decision Threshold | Macro-Averaged Validation $F_{0.5}$ |
|:---:|:---:|
| 0.20 | 0.8298 |
| 0.30 | 0.8448 |
| 0.40 | 0.8522 |
| 0.50 | 0.8560 |
| 0.55 | 0.8570 |
| 0.58 | 0.8574 |
| 0.59 | 0.8575 |
| **0.60** | **0.8575 (Optimal)** |
| 0.61 | 0.8575 |
| 0.65 | 0.8573 |
| 0.70 | 0.8565 |
| 0.80 | 0.8523 |

At threshold **`0.60`**, the classifier achieves:
- **Pairwise Precision**: **97.90%** ($828,765$ TP vs. only $17,752$ FP)
- **Pairwise Recall**: **94.09%** ($52,029$ FN)
- **Pairwise $F_{0.5}$**: **97.12%**
- **Macro-Averaged Precision**: **92.39%**
- **Macro-Averaged Recall**: **74.03%**
- **Macro-Averaged $F_{0.5}$ (Leaderboard Metric)**: **0.8575** (up from initial baseline of **0.8569**)

The tuned value `0.60` is automatically saved to `models/best_threshold.txt` and loaded during test inference.

---

## 7. Engineering Ablations & Key Discoveries

During research and development, we tested multiple configurations on the real dataset to identify the optimal performance-recall frontier:

| Iteration / Hypothesis | Configuration | Observed Outcome | Action Taken |
|---|---|---|---|
| **Ablation 1: Char N-Gram Blocking** | `char_wb (3,5)`, 100K feat, `max_df=0.05`, 2 separate passes | Matrix density reached 35 non-zeros/row. India blocking alone projected at **3.5 hours**; full blocking projected at **17+ hours**. | **REJECTED**: Switched to word-level TF-IDF. |
| **Ablation 2: Word Unigrams vs. Bigrams** | Word `(1,1)`, 30K feat, `max_df=0.005` | Vectorization was extremely fast (6 min), but density dropped to 5.0 non-zeros/row. Missed compound brand names like "apple computer". | **REJECTED**: Bigrams provide critical multi-word context. |
| **Ablation 3: Unified Name+Address Blocking** | Word `(1,2)`, 30K feat, `max_df=0.005`, single pass | Blocking time dropped to **28.3 minutes** across 22.06M pairs while preserving >94% candidate recall. | **ADOPTED**: Core production blocking engine. |
| **Ablation 4: Default Threshold (0.50)** | Standard LightGBM default $P \ge 0.50$ | Achieved $F_{0.5} = 0.8556$. Suffered from subtle false merges on edge-case singletons (reducing entity score to 0.0). | **REJECTED**: Tuned higher to maximize precision. |
| **Ablation 5: $F_{0.5}$-Tuned Threshold (0.60)** | Swept threshold $P \ge 0.60$ (17 features) | Increased pair precision to **97.89%**, boosting macro-averaged $F_{0.5}$ to **0.8569**. | **SUPERSEDED**: Baseline model. |
| **Ablation 6: 21 Features + Fine-Grained Thresholding** | 21 features (added token containment, exact name match, PIN match/mismatch) + $0.01$ step threshold sweep | Validation logloss dropped to **0.0542**. Macro-averaged $F_{0.5}$ improved to **0.8575** (+0.0006 gain). Zero validator issues. | **ADOPTED**: Core feature engineering. |
| **Ablation 7: N-to-1 Conflict Resolution Post-Processing** | Enforce 1-to-1 candidate constraint (assign candidate ID exclusively to the $S_1$ entity with highest $P$) | Pruned 37,489 multi-claimed candidate false positives. Macro-averaged $F_{0.5}$ improved from **0.8573** to **0.8577** without modifying model weights. | **ADOPTED**: Final production submission. |

---

## 8. Hardware, Scalability & Resource Utilization

The entire pipeline is engineered for extreme cost and resource efficiency without requiring high-cost GPU infrastructure:

| Phase | Runtime | Resource Utilization | Memory Footprint |
|---|---|---|---|
| **Phase 1: Preprocessing** | ~15 min (cached) | 16 CPU cores (vectorized list operations) | ~6 GB peak RAM |
| **Phase 2: Blocking** | **28.3 min** | 16 CPU threads (`sparse_dot_topn` OpenMP) | ~8 GB peak RAM (200K chunking) |
| **Phase 3: Features** | **4.0 min** | 14 CPU cores (multiprocessing worker pool) | ~10 GB peak RAM (streaming parquet) |
| **Phase 4: Training & Tuning** | **3.5 min** | 16 CPU cores (OpenMP LightGBM) | ~11 GB peak RAM (3:1 sampled) |
| **End-to-End Pipeline** | **~36 min** | Standard commodity CPU workstation | **< 12 GB peak RAM** |

---

## 9. Submission Verification & Output Compliance

The inference pipeline (`pipeline.py --test`) processes `test_source1.tsv` (1,732,544 rows), `test_source2.tsv` (4,887,273 rows), and `test_source3.tsv` (5,082,316 rows).

### Validator Verification
Running the official `utils/validate_submission.py` with `--check-ids` against all test source IDs produces:
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1,732,544
  valid S2/S3 match IDs: 9,969,589
  matching_results.tsv: 1,732,544 rows (186,297 empty, 1,546,247 non-empty).
  candidate_pairs.tsv:  1,732,544 rows (390 empty, 1,732,154 non-empty).

PASS — no blocking issues found. Safe to submit.
```
- Exactly one row per Source 1 entity in both files (1,732,544 rows).
- Singletons correctly represented by empty strings.
- Matching results are a verified 100% subset of candidate pairs.
- Zero ID duplication and zero invalid IDs.

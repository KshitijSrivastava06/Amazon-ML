# Entity Resolution Methodology Document

## 1. High-Level Methodology

Our solution delivers an end-to-end, high-performance, and open-set supervised machine learning pipeline engineered specifically for the Amazon ML Challenge 2026: Business Entity Resolution. The system resolves noisy, multilingual business records across three independent data sources at extreme scale (~24.2M total records) while strictly adhering to all academic integrity, licensing, and competition guidelines.

### Core Architectural Principles
1. **Open-Set Generalization (Zero Country Hardcoding)**: In accordance with the problem statement, `country` is treated as an open set of string labels. The pipeline contains zero country-specific branches or one-hot encodings. Training data covers only `US` and `India`, yet the pipeline processes `France` (and any other arbitrary country) in the test set without modification. Country serves strictly as an independent partition key for candidate generation (since entity boundaries are contained within national jurisdictions).
2. **Zero External Lookups (Strict Academic Fair Play)**: No external databases, commercial ER APIs, geocoders, or web queries are used. All signals are learned and computed strictly from the provided data.
3. **Extreme Scale Optimization**: With 2.2M Source 1 records and over 10.3M Source 2/Source 3 records, brute-force comparison ($O(N \cdot M) \approx 2.3 \times 10^{13}$ pairs) is impossible. Our pipeline utilizes an optimized word-level sparse candidate generator, streaming disk writes, chunked multi-core feature extraction, and gradient-boosted decision trees to complete the end-to-end pipeline in **~36 minutes** on a standard multi-core machine.
4. **$F_{0.5}$ Metric Alignment**: The competition metric weights precision twice as heavily as recall ($\beta = 0.5$). Merging two distinct businesses (false positive) causes severe downstream commercial damage. Our decision boundary is strictly optimized for macro-averaged $F_{0.5}$, effectively capturing genuine duplicates while maintaining high precision on singletons.

---

## 2. Universal Preprocessing & Normalization

To handle multilingual scripts, legal abbreviations, and typographic errors across heterogeneous platforms without external lookups, all text undergoes a deterministic, high-speed normalization pipeline:

1. **Phonetic ASCII Transliteration (`anyascii`)**:
   - Indian records in Source 2/3 frequently feature non-Latin scripts (e.g., Devanagari: `राम मार्केटिंग प्राइवेट लिमिटेड`) while Source 1 uses transliterated English (`Ram Marketing Private Limited`).
   - French records in the test set contain accented European characters (`SCI Ptit Àmicale`, `Café de la Mairie`, `École Supérieure`).
   - `anyascii` converts all non-ASCII unicode scripts into phonetically consistent standard Latin text offline without network calls.
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

Every candidate pair $(S_1, S_{2/3})$ is enriched with an exhaustive suite of **17 language-agnostic similarity features** computed in parallel across CPU cores using the C++ backend of `rapidfuzz`. 

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

Features are computed using multiprocessing chunks and streamed directly to disk as `train_features.parquet` / `test_features.parquet`, achieving a throughput of **>88,000 pairs/sec**.

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
- **Result**: Validation binary logloss dropped steadily from **0.3214** (round 10) to **0.0549** (round 500).

### 5.3 Top Feature Importance (Gain)
The model relies overwhelmingly on brand prefixes and structural address tokens:
1. `name_jaro_winkler` ($7.34 \times 10^7$ gain) — Primary brand alignment
2. `addr_shared_num` ($2.70 \times 10^7$ gain) — PIN code and street number agreement
3. `addr_token_set` ($2.03 \times 10^7$ gain) — Address locality and city overlap
4. `name_token_set` ($1.10 \times 10^7$ gain) — Word set overlap
5. `name_len_ratio` ($9.57 \times 10^6$ gain) — Name brevity agreement

---

## 6. Metric Optimization & $F_{0.5}$ Threshold Tuning

The competition evaluates submissions using **Macro-Averaged $F_{0.5}$**, calculated per $S_1$ entity and averaged across all entities (including singletons).

### 6.1 Formula Alignment
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Because $F_{0.5}$ penalizes false positives twice as heavily as false negatives, a default threshold of $0.50$ is sub-optimal. Furthermore, correctly outputting an empty string for a singleton entity awards a full **1.0** credit, while predicting a single spurious match on a singleton reduces its score to **0.0**.

### 6.2 Empirical Threshold Sweep
Post-training, we simulate the exact leaderboard evaluation on all **3,309,804 validation pairs** across validation entities:

| Decision Threshold | Macro-Averaged Validation $F_{0.5}$ |
|:---:|:---:|
| 0.20 | 0.8288 |
| 0.30 | 0.8440 |
| 0.40 | 0.8517 |
| 0.50 | 0.8556 |
| 0.55 | 0.8565 |
| **0.60** | **0.8569 (Optimal)** |
| 0.65 | 0.8566 |
| 0.70 | 0.8558 |
| 0.80 | 0.8514 |

At threshold **`0.60`**, the classifier achieves:
- **Pair Precision**: **97.89%** ($827,505$ TP vs. only $17,835$ FP)
- **Pair Recall**: **93.95%** ($53,289$ FN)
- **Pair $F_{0.5}$**: **0.9708**
- **Macro-Averaged $F_{0.5}$**: **0.8569**

The tuned value `0.60` is automatically saved to `models/best_threshold.txt` and loaded during test inference.

---

## 7. Submission Verification & Output Compliance

The inference pipeline (`pipeline.py --test`) processes `test_source1.tsv` (1,732,544 rows), `test_source2.tsv` (4,887,273 rows), and `test_source3.tsv` (5,082,316 rows).

### Validator Verification
Running the official `utils/validate_submission.py` with `--check-ids` against all test source IDs produces:
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1,732,544
  valid S2/S3 match IDs: 9,969,589
  matching_results.tsv: 1,732,544 rows (183,101 empty, 1,549,443 non-empty).
  candidate_pairs.tsv:  1,732,544 rows (390 empty, 1,732,154 non-empty).

PASS — no blocking issues found. Safe to submit.
```
- Exactly one row per Source 1 entity in both files (1,732,544 rows).
- Singletons correctly represented by empty strings.
- Matching results are a verified 100% subset of candidate pairs.
- Zero ID duplication and zero invalid IDs.

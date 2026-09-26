# Entity Resolution Methodology Document

## 1. High-Level Methodology
Our solution employs a robust, two-stage supervised machine learning pipeline designed to handle extreme scale and high noise variance across multiple languages.

The fundamental design principle of this pipeline is that **Country is treated as an open set of string labels**. No country-specific logic (e.g., hardcoded state abbreviations or rule branches) is used. The pipeline dynamically partitions data by whatever country values exist and applies universal string-similarity operations, allowing it to seamlessly generalize to unseen countries like France.

The pipeline consists of:
1.  **Universal Preprocessing**: Lowercasing, punctuation stripping, and transliteration using `anyascii` (which converts Devanagari script and French accents to standard Latin representations without API calls). Universal abbreviation expansion normalizes business types (e.g., Pvt->private, SARL->societe a responsabilite limitee).
2.  **TF-IDF Blocking (Candidate Generation)**: To reduce the O(N²) search space, we compute character n-gram TF-IDF vectors for both names and addresses, grouped by country partition. We use `sparse_dot_topn` to rapidly find the top 20 name and top 10 address candidates.
3.  **Feature Engineering**: We extract 17 numeric string-similarity features per candidate pair using the highly optimized `rapidfuzz` library (C++ backend).
4.  **Classification & Thresholding**: A LightGBM binary classifier learns the probability of a match. We then dynamically sweep prediction thresholds to maximize the competition's macro-averaged F0.5 metric, heavily favoring precision.

## 2. Blocking / Candidate Generation Strategy
Brute-force pairwise comparison of ~2M S1 records against ~10M S2/S3 records is computationally infeasible. Our blocking strategy safely reduces the candidate space while maintaining a >95% recall ceiling.

*   **Dynamic Country Partitioning**: The algorithm discovers all unique values in the `country` column at runtime and processes each group independently. Since businesses do not cross country borders in this dataset, this trivially cuts the search space without risking recall.
*   **Dual Character N-gram TF-IDF**: We train TF-IDF vectorizers using `analyzer='char_wb'` and `ngram_range=(3,5)`. Character n-grams naturally capture similarities despite typos, transliteration errors, and missing words.
*   **Dual Pass (Name & Address)**:
    *   **Name Pass**: Finds the top 20 most similar S2/S3 names for each S1 name (minimum cosine similarity 0.25).
    *   **Address Pass**: Finds the top 10 most similar S2/S3 addresses for each S1 address (minimum cosine similarity 0.20).
*   **Union Strategy**: The final candidate set for a given S1 entity is the union of the candidates found in the name pass and the address pass. This ensures that if a record has a terribly misspelled name but a clean address (or vice versa), it is still captured.

## 3. Feature Engineering
For every candidate pair generated in the blocking stage, we compute a comprehensive suite of 17 language-agnostic string similarity metrics. No country metadata is used as a feature, forcing the model to generalize purely on string patterns.

**Name Features:**
*   `rapidfuzz.fuzz.ratio` (Normalized Levenshtein edit distance)
*   `rapidfuzz.distance.JaroWinkler.similarity` (Heavily weights matching prefixes)
*   `rapidfuzz.fuzz.token_sort_ratio` (Handles out-of-order words, e.g., "Apple Corp" vs "Corp Apple")
*   `rapidfuzz.fuzz.token_set_ratio` (Handles missing/extra tokens)
*   `rapidfuzz.fuzz.partial_ratio` (Identifies when one string is a substring of the other, useful for DBA/trade names)
*   `name_jaccard` (Word-level intersection over union)
*   `name_len_ratio` (Ratio of shorter name length to longer name length)

**Address Features:**
*   `addr_levenshtein`, `addr_token_sort`, `addr_token_set`, `addr_jaccard`
*   `addr_shared_num` (Calculates the intersection over maximum length of purely numeric tokens, effectively matching PIN codes and street numbers regardless of surrounding text)
*   `addr_both_null` / `addr_one_null` (Binary flags to handle the ~3.4% of S2/S3 records missing addresses)

**Cross Features:**
*   `cross_n1_a2` and `cross_n2_a1` (Partial ratios comparing the Name of Source 1 to the Address of Source 2/3. This catches data-entry errors where the business name is stuffed into the address field).

## 4. Model Architecture
We selected **LightGBM** (MIT License, << 8B parameters) for the classification stage.

*   **Training Setup**: Candidate pairs are labeled using the ground truth. We apply a 3:1 negative sampling ratio (3 negatives per positive) to maintain a healthy class balance for training.
*   **Group-wise Splitting**: We use `GroupShuffleSplit` on `source1_entity_id` to create an 85/15 Train/Validation split. Grouping by S1 entity ensures no data leakage (all pairs belonging to a single S1 entity stay together).
*   **Hyperparameters**: We use 127 leaves, a learning rate of 0.05, and `is_unbalance=True`. Early stopping is applied on the validation logloss.
*   **Threshold Tuning**: Because the F0.5 score weights precision 2x over recall, the standard 0.5 probability cutoff is rarely optimal. Post-training, we sweep thresholds from 0.1 to 0.9 on the validation set, calculating the macro-averaged F0.5 score at each step (including scoring singletons exactly as the public leaderboard does). The threshold that strictly maximizes F0.5 is saved and applied during test inference.

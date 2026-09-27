# Business Entity Resolution: Comprehensive Dataset Guide

## 1. Overview & Objective

The **Amazon ML Challenge 2026: Business Entity Resolution** focuses on resolving entity identities across multiple noisy, heterogeneous data sources. In enterprise e-commerce and commercial databases, business entity records arrive asynchronously from independent providers without shared global keys. 

The goal is to determine which records from **Source 2** and **Source 3** represent the exact same real-world business entity as a reference record in **Source 1**.

```
+---------------------------+
| Source 1 (Reference Gold) | <--- Deduplicated anchor entities (2.2M train, 1.7M test)
+---------------------------+
              |
              | Finds matching records across sources
              v
+---------------------------+        +---------------------------+
|   Source 2 (Uncurated)    |  and   |   Source 3 (Uncurated)    |
| (5.0M train, 4.9M test)   |        | (5.3M train, 5.1M test)   |
+---------------------------+        +---------------------------+
```

---

## 2. Dataset Scale & Specifications

The total corpus contains **~24.2 million records** across 7 primary `.tsv` files.

### 2.1 File Inventory & Statistics

| Dataset Split | File Name | Record Count | File Size | Null Names | Null Addresses | Primary Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | [`train_source1.tsv`](file:///c:/Users/koolr/Desktop/Anything/Amazon%20ML/student_resource/dataset/train/train_source1.tsv) | **2,206,821** | ~200 MB | 0 | 0 | Anchor reference records |
| **Train** | [`train_source2.tsv`](file:///c:/Users/koolr/Desktop/Anything/Amazon%20ML/student_resource/dataset/train/train_source2.tsv) | **5,034,616** | ~467 MB | 2 | 168,967 (3.4%) | Secondary uncurated records |
| **Train** | [`train_source3.tsv`](file:///c:/Users/koolr/Desktop/Anything/Amazon%20ML/student_resource/dataset/train/train_source3.tsv) | **5,285,603** | ~480 MB | 13 | 175,916 (3.3%) | Secondary uncurated records |
| **Train** | [`train_ground_truth.tsv`](file:///c:/Users/koolr/Desktop/Anything/Amazon%20ML/student_resource/dataset/train/train_ground_truth.tsv) | **2,206,821** | ~121 MB | — | — | Ground truth match mappings |
| **Test** | [`test_source1.tsv`](file:///c:/Users/koolr/Desktop/Anything/Amazon%20ML/student_resource/dataset/test/test_source1.tsv) | **1,732,544** | ~167 MB | 0 | 0 | Test anchor entities to resolve |
| **Test** | [`test_source2.tsv`](file:///c:/Users/koolr/Desktop/Anything/Amazon%20ML/student_resource/dataset/test/test_source2.tsv) | **4,887,273** | ~486 MB | 46 | 129,408 (2.6%) | Candidate pool |
| **Test** | [`test_source3.tsv`](file:///c:/Users/koolr/Desktop/Anything/Amazon%20ML/student_resource/dataset/test/test_source3.tsv) | **5,082,316** | ~483 MB | 59 | 136,098 (2.7%) | Candidate pool |

> **Format Constraint:** All files are strictly tab-separated (`sep='\t'`). Commas appear naturally inside business addresses and match lists, so reading with comma delimiters will corrupt row parsing.

---

## 3. Schema & Field Definitions

Each record table (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) shares an identical four-column schema:

| Column Name | Data Type | Description & Characteristics |
| :--- | :--- | :--- |
| `entity_id` | `string` | Unique record identifier prefixed by source: `S1-xxxxx`, `S2-xxxxx`, or `S3-xxxxx`. |
| `business_name` | `string` | Name of the business. Contains legal suffixes, abbreviations, noise prefixes, non-Latin scripts, and typos. |
| `business_address` | `string` | Physical address. Highly variable formatting, missing components (e.g., missing PIN or state), landmark references, or `NaN`. |
| `country` | `string` | Country label (`US`, `India`, `France`). **Partition key only.** |

### Ground Truth Schema (`train_ground_truth.tsv`)

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `source1_entity_id` | `string` | The anchor S1 entity identifier. | `S1-00042` |
| `matched_entity_ids` | `string` | Comma-separated list of matching S2/S3 IDs. **Empty string for singletons.** | `S2-10821,S3-00412` |

---

## 4. Geographic Distribution & Open-Set Generalization

One of the central design challenges is **Open-Set Domain Shift**.

### Geographic Breakdown Across Splits

```
[TRAIN SET: 2.2M S1 records]          [TEST SET: 1.7M S1 records]
+-------------------+                 +-------------------+
|  US: 60.0%        |                 |  India: 46.8%     |
|  India: 40.0%     |                 |  US: 38.3%        |
+-------------------+                 |  France: 15.0% 🚀 | <--- Unseen Country!
                                      +-------------------+
```

| Country | Training S1 | Training S2 | Training S3 | Test S1 | Test S2 | Test S3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **United States (`US`)** | 1,323,633 (60.0%) | 3,016,817 (59.9%) | 3,170,056 (60.0%) | 663,106 (38.3%) | 1,871,330 (38.3%) | 1,945,701 (38.3%) |
| **India (`India`)** | 883,188 (40.0%) | 2,017,799 (40.1%) | 2,115,547 (40.0%) | 809,986 (46.8%) | 2,312,565 (47.3%) | 2,405,000 (47.3%) |
| **France (`France`)** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **259,452 (15.0%)** | **703,378 (14.4%)** | **731,615 (14.4%)** |

### Crucial Engineering Implications:
1. **Zero Hardcoded Logic:** No code should contain `if country == 'US'` or country-specific state abbreviation dictionaries.
2. **No Country Features in Classifier:** Country is **not** passed as a feature to LightGBM. Features are purely string similarity measures so the classifier generalizes universally.
3. **Partitioning Key:** Businesses do not cross country borders in this dataset. Country serves strictly as an independent partition for blocking.

---

## 5. Ground Truth Linkage & Singleton Dynamics

Analysis of `train_ground_truth.tsv` reveals critical structural properties of the entity linkages:

```
Match Count Distribution per S1 Entity:
  0 matches (Singletons) :   123,247 entities ( 5.6%)  <--- Crucial for metric
  1 match                :   141,832 entities ( 6.4%)
  2 matches              :   289,114 entities (13.1%)
  3 matches              :   512,408 entities (23.2%)
  4 matches (Median)     :   680,210 entities (30.8%)
  5+ matches             :   459,990 entities (20.9%)
  Maximum matches        :   11 matches
  Mean matches per S1    :   3.67 records
```

### The Singleton Factor:
- **5.6% of S1 entities are singletons** — they have **zero** matching records in Source 2 or Source 3.
- In the evaluation metric, correctly outputting an empty prediction (`""`) for a true singleton scores **1.0**. 
- Conversely, predicting even a single false match for a singleton drops that entity's score immediately to **0.0**.

---

## 6. Real-World Noise Patterns & Anomalies

The dataset mimics commercial web-scraped and user-entered entity data, exhibiting high noise variance:

### 6.1 Multilingual & Script Mismatches
- **Devanagari Script:** In Indian records, Source 2 and Source 3 frequently contain Devanagari script names (`राम मार्केटिंग प्राइवेट लिमिटेड`) while Source 1 contains transliterated English (`Ram Marketing Private Limited`).
- **Accented European Characters:** French records in the test set contain accents (`SCI Ptit Àmicale`, `Café de la Mairie`, `École Supérieure`).
- **Solution:** Universal ASCII transliteration (`anyascii`) converts characters phonetically into standard Latin characters before matching.

### 6.2 Legal Suffix & Corporate Abbreviation Noise
Entities exhibit extensive corporate suffix variations:
- English: `Corp` ↔ `Corporation`, `Pvt Ltd` ↔ `Private Limited`, `Inc` ↔ `Incorporated`, `Co.` ↔ `Company`.
- French: `SARL` ↔ `Société à responsabilité limitée`, `SAS` ↔ `Société par actions simplifiée`, `SCI` ↔ `Société civile immobilière`.
- Formatting symbols: `&` ↔ `and`, `+` ↔ `plus`.

### 6.3 Missing Data & Null Addresses
- Source 1 has zero null values.
- Source 2 and Source 3 have **~3.4% missing addresses** (`NaN` / `nan`).
- For records with null addresses, blocking and matching must rely entirely on name similarities, handled via explicit `addr_both_null` and `addr_one_null` indicator features.

### 6.4 Address Structural Inconsistencies
- **Landmark-based Addressing:** Common in Indian addresses (`Near SBI Bank, Opp Municipal Market, M.G. Road`).
- **PIN Code Variations:** Zip codes embedded in different positions, truncated, or absent.
- **Component Reordering:** `Street, City, State, PIN` vs `PIN, City, Street`.
- **Numeric Token Significance:** Door numbers and postal codes are captured via numeric token overlap ratios (`addr_shared_num`).

---

## 7. Leaderboard Metric: Macro-Averaged $F_{0.5}$

Submissions are evaluated on the **Macro-Averaged $F_{0.5}$ score** across all Source 1 entities in the test set.

### 7.1 Mathematical Definition

For each S1 entity $i$:

$$\text{Precision}_i = \frac{|T_i \cap P_i|}{|P_i|}, \quad \text{Recall}_i = \frac{|T_i \cap P_i|}{|T_i|}$$

$$F_{0.5, i} = \frac{(1 + 0.5^2) \times \text{Precision}_i \times \text{Recall}_i}{0.5^2 \times \text{Precision}_i + \text{Recall}_i} = \frac{1.25 \times \text{Precision}_i \times \text{Recall}_i}{0.25 \times \text{Precision}_i + \text{Recall}_i}$$

$$\text{Final Score} = \frac{1}{N} \sum_{i=1}^{N} F_{0.5, i}$$

### 7.2 Why $F_{0.5}$ Dictates Model Strategy
- **Precision is weighted 2× as heavily as Recall:** False positives (false merges) penalize the score far more severely than false negatives (missed links).
- In entity resolution, accidentally linking two distinct businesses creates catastrophic business errors (mixing financial or catalog data). The metric reflects this by heavily penalizing over-merging.
- **Optimal Decision Threshold:** In our classification stage, the decision threshold must be tuned higher (typically $0.65 - 0.80$) rather than default $0.50$ to maximize precision.

### 7.3 Edge Cases & Singleton Scoring Matrix

| True Set ($T_i$) | Predicted Set ($P_i$) | Entity Score |
| :---: | :---: | :---: |
| $\emptyset$ (Singleton) | $\emptyset$ | **1.0** (Correctly identified singleton) |
| $\emptyset$ (Singleton) | Non-empty | **0.0** (False merge) |
| Non-empty | $\emptyset$ | **0.0** (Completely missed entity) |
| Non-empty | Partial match | Calculated via $F_{0.5}$ formula |
| Non-empty | No overlap ($|T_i \cap P_i| = 0$) | **0.0** |

---

## 8. Required Output Specifications

Your trained inference pipeline generates two tab-separated files inside `output/`:

### 8.1 `output/matching_results.tsv` (Scored File)
- **Format:** Exactly two columns separated by a tab (`\t`): `source1_entity_id` and `matched_entity_ids`.
- **Requirement:** Must contain exactly **1,732,544 rows** (one per S1 entity in `test_source1.tsv`).
- **Syntax:**
  ```tsv
  source1_entity_id	matched_entity_ids
  S1-100001	S2-302194,S3-401928
  S1-100002	S2-501923
  S1-100003	
  ```

### 8.2 `output/candidate_pairs.tsv` (Audit File)
- **Format:** Exactly two columns separated by a tab (`\t`): `source1_entity_id` and `candidate_entity_ids`.
- Represents the high-recall pool generated by the blocking stage before machine learning classification.
- Guarantees that every match in `matching_results.tsv` is a verified subset of `candidate_pairs.tsv`.

"""
Configuration for the Business Entity Resolution Pipeline.
All paths, hyperparameters, and constants in one place.

DESIGN PRINCIPLE: Country is treated as an OPEN SET of string labels.
- Country is used ONLY as a partitioning key for blocking (same-country matching).
- ALL preprocessing, features, and model logic are country-agnostic.
- No country-specific code paths (no `if country == "X"` branches).
- Abbreviation maps are UNIVERSAL — they apply to all records regardless of country.
"""
import os
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_DIR = os.environ.get('DATASET_DIR', os.path.join(BASE_DIR, 'dataset'))
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(BASE_DIR, 'output'))
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, 'code', 'business_entity_resolution', 'models'))

# Training files
TRAIN_S1 = os.path.join(TRAIN_DIR, 'train_source1.tsv')
TRAIN_S2 = os.path.join(TRAIN_DIR, 'train_source2.tsv')
TRAIN_S3 = os.path.join(TRAIN_DIR, 'train_source3.tsv')
TRAIN_GT = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

# Test files
TEST_S1 = os.path.join(TEST_DIR, 'test_source1.tsv')
TEST_S2 = os.path.join(TEST_DIR, 'test_source2.tsv')
TEST_S3 = os.path.join(TEST_DIR, 'test_source3.tsv')

# Output files
MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

def update_paths(custom_dataset_dir=None, custom_output_dir=None, custom_model_dir=None):
    """Dynamically updates paths for cloud environments like Kaggle or Google Colab."""
    global DATASET_DIR, TRAIN_DIR, TEST_DIR
    global TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
    global TEST_S1, TEST_S2, TEST_S3
    global OUTPUT_DIR, MATCHING_OUTPUT, CANDIDATE_OUTPUT
    global MODEL_DIR

    if custom_dataset_dir:
        DATASET_DIR = os.path.abspath(custom_dataset_dir)
        # Search both root dataset_dir and subdirectories train/ and test/
        def resolve(subfolder, fname):
            c1 = os.path.join(DATASET_DIR, subfolder, fname)
            c2 = os.path.join(DATASET_DIR, fname)
            return c1 if os.path.exists(c1) else (c2 if os.path.exists(c2) else c1)
        
        TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
        TEST_DIR = os.path.join(DATASET_DIR, 'test')
        TRAIN_S1 = resolve('train', 'train_source1.tsv')
        TRAIN_S2 = resolve('train', 'train_source2.tsv')
        TRAIN_S3 = resolve('train', 'train_source3.tsv')
        TRAIN_GT = resolve('train', 'train_ground_truth.tsv')
        TEST_S1 = resolve('test', 'test_source1.tsv')
        TEST_S2 = resolve('test', 'test_source2.tsv')
        TEST_S3 = resolve('test', 'test_source3.tsv')

    if custom_output_dir:
        OUTPUT_DIR = os.path.abspath(custom_output_dir)
        MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
        CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

    if custom_model_dir:
        MODEL_DIR = os.path.abspath(custom_model_dir)

# ─── Preprocessing ───────────────────────────────────────────────────
# Universal business name abbreviation mappings (language-agnostic).
# Applied to ALL records regardless of country.
BUSINESS_ABBREVIATION_MAP = {
    # English legal suffixes
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bltd\b': 'limited',
    r'\bpvt\b': 'private',
    r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership',
    r'\bco\b': 'company',
    # French legal suffixes
    r'\bsarl\b': 'societe a responsabilite limitee',
    r'\bsas\b': 'societe par actions simplifiee',
    r'\bsa\b': 'societe anonyme',
    r'\bsci\b': 'societe civile immobiliere',
    r'\beurl\b': 'entreprise unipersonnelle a responsabilite limitee',
    r'\bsnc\b': 'societe en nom collectif',
    r'\bsei\b': 'societe en participation',
    # Common business terms (universal)
    r'\bsvcs\b': 'services',
    r'\bsvc\b': 'service',
    r'\bmfg\b': 'manufacturing',
    r'\bintl\b': 'international',
    r'\bntnl\b': 'national',
    r'\bassoc\b': 'associates',
    r'\bassn\b': 'association',
    r'\btech\b': 'technology',
    r'\bengr\b': 'engineering',
    r'\bengg\b': 'engineering',
    r'\bhosp\b': 'hospital',
    r'\bpharm\b': 'pharmaceutical',
    r'\bindust\b': 'industries',
    r'\bind\b': 'industries',
    r'\bentpr\b': 'enterprises',
    r'\benterp\b': 'enterprises',
    r'\bgrp\b': 'group',
    r'\bfdn\b': 'foundation',
    r'\bfound\b': 'foundation',
    r'\bcomm\b': 'communications',
    r'\bsol\b': 'solutions',
    r'\bmgmt\b': 'management',
    r'\bdist\b': 'distributors',
    r'\bhldgs\b': 'holdings',
    r'\b&\b': 'and',
    r'\bet\b': 'and',           # French "and"
}

# Universal address abbreviation mappings (language-agnostic).
# Covers common abbreviations from multiple address systems.
ADDRESS_ABBREVIATION_MAP = {
    # Road/street types (universal)
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    # Building/unit types (universal)
    r'\bfl\b': 'floor',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bbldg\b': 'building',
    r'\bno\b': 'number',
    # French address types
    r'\br\b': 'rue',
    r'\bbd\b': 'boulevard',
    r'\bav\b': 'avenue',
    # Directional (universal)
    r'\bn\b': 'north',
    r'\bs\b': 'south',
    r'\be\b': 'east',
    r'\bw\b': 'west',
    r'\bnw\b': 'northwest',
    r'\bne\b': 'northeast',
    r'\bsw\b': 'southwest',
    r'\bse\b': 'southeast',
    # District/area (universal)
    r'\btq\b': 'taluk',
    r'\bdist\b': 'district',
    r'\bvlg\b': 'village',
    r'\bcol\b': 'colony',
}

# ─── Blocking / Candidate Generation ─────────────────────────────────
# TF-IDF parameters for name-based blocking
TFIDF_NAME_PARAMS = {
    'analyzer': 'char_wb',
    'ngram_range': (3, 5),
    'max_features': 100000,
    'min_df': 3,          # Drop n-grams appearing in fewer than 3 docs (noise)
    'max_df': 0.3,        # Drop n-grams appearing in >30% of docs (stop-n-grams)
    'sublinear_tf': True,
    'dtype': np.float32,
}

# TF-IDF parameters for address-based blocking
TFIDF_ADDR_PARAMS = {
    'analyzer': 'char_wb',
    'ngram_range': (3, 5),
    'max_features': 80000,
    'min_df': 3,          # Drop n-grams appearing in fewer than 3 docs (noise)
    'max_df': 0.3,        # Drop n-grams appearing in >30% of docs (stop-n-grams)
    'sublinear_tf': True,
    'dtype': np.float32,
}

# sparse_dot_topn: number of top candidates per S1 entity
BLOCKING_TOP_K_NAME = 10
BLOCKING_TOP_K_ADDR = 10

# Minimum cosine similarity threshold during blocking
BLOCKING_THRESHOLD_NAME = 0.40
BLOCKING_THRESHOLD_ADDR = 0.30

# ─── LightGBM ────────────────────────────────────────────────────────
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'n_estimators': 500,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
    'is_unbalance': True,
}

# Negative sampling ratio (negatives per positive) for training
NEG_SAMPLE_RATIO = 3

# Validation split ratio
VAL_SPLIT_RATIO = 0.15

# Matching threshold (will be tuned for F_0.5)
MATCH_THRESHOLD = 0.5

# ─── Misc ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import time
import os

def compute_jaccard_similarity(s1, s2):
    """Word-level Jaccard similarity."""
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return 0.0
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_shared_numeric_ratio(s1, s2):
    """Calculates the ratio of shared numeric tokens (useful for addresses/PINs)."""
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return 0.0
        
    num1 = set([w for w in s1.split() if any(c.isdigit() for c in w)])
    num2 = set([w for w in s2.split() if any(c.isdigit() for c in w)])
    
    if not num1 and not num2:
        return 0.0 # Neither has numbers
    if not num1 or not num2:
        return 0.0 # One has numbers, the other doesn't
        
    return len(num1.intersection(num2)) / max(len(num1), len(num2))

def extract_features_for_pair(row):
    """
    Computes all string similarity features for a single candidate pair.
    Designed to work with pandas apply on a DataFrame row.
    """
    n1 = str(row['name_clean_1'])
    n2 = str(row['name_clean_2'])
    a1 = str(row['addr_clean_1'])
    a2 = str(row['addr_clean_2'])
    
    # Check if either string is purely "nan" string which shouldn't happen with our preprocessing, 
    # but just in case.
    if n1 == 'nan': n1 = ''
    if n2 == 'nan': n2 = ''
    if a1 == 'nan': a1 = ''
    if a2 == 'nan': a2 = ''

    features = {}
    
    # --- 1. Name Features ---
    # rapidfuzz returns 0-100. We normalize to 0-1.
    features['name_levenshtein'] = fuzz.ratio(n1, n2) / 100.0
    features['name_jaro_winkler'] = JaroWinkler.similarity(n1, n2)
    features['name_token_sort'] = fuzz.token_sort_ratio(n1, n2) / 100.0
    features['name_token_set'] = fuzz.token_set_ratio(n1, n2) / 100.0
    features['name_partial'] = fuzz.partial_ratio(n1, n2) / 100.0
    features['name_jaccard'] = compute_jaccard_similarity(n1, n2)
    
    # Length ratios
    len_n1, len_n2 = len(n1), len(n2)
    features['name_len_ratio'] = min(len_n1, len_n2) / max(len_n1, len_n2) if max(len_n1, len_n2) > 0 else 0.0
    
    # --- 2. Address Features ---
    if a1 and a2:
        features['addr_levenshtein'] = fuzz.ratio(a1, a2) / 100.0
        features['addr_token_sort'] = fuzz.token_sort_ratio(a1, a2) / 100.0
        features['addr_token_set'] = fuzz.token_set_ratio(a1, a2) / 100.0
        features['addr_jaccard'] = compute_jaccard_similarity(a1, a2)
        features['addr_shared_num'] = compute_shared_numeric_ratio(a1, a2)
        features['addr_both_null'] = 0
        features['addr_one_null'] = 0
    else:
        features['addr_levenshtein'] = 0.0
        features['addr_token_sort'] = 0.0
        features['addr_token_set'] = 0.0
        features['addr_jaccard'] = 0.0
        features['addr_shared_num'] = 0.0
        features['addr_both_null'] = 1 if not a1 and not a2 else 0
        features['addr_one_null'] = 1 if (not a1 and a2) or (a1 and not a2) else 0

    # --- 3. Cross Features ---
    # Sometimes business names are buried in addresses, or vice versa
    features['cross_n1_a2'] = fuzz.partial_ratio(n1, a2) / 100.0 if a2 else 0.0
    features['cross_n2_a1'] = fuzz.partial_ratio(n2, a1) / 100.0 if a1 else 0.0
    
    # --- 4. Meta Features ---
    # Note: We intentionally do NOT use country as a feature.
    # Model must generalize purely based on string similarity.
    features['is_source_2'] = 1 if row['entity_id_2'].startswith('S2-') else 0
    
    return pd.Series(features)

def build_feature_matrix(candidates_df, s1_df, s2_df, s3_df):
    """
    Takes candidate pairs, joins the text data, and computes features.
    
    candidates_df expects columns: ['source1_entity_id', 'candidate_entity_id']
    (Note: candidate_pairs.tsv has a comma-separated list, this function expects 
    it to be exploded so there is one row per pair).
    """
    print(f"Building feature matrix for {len(candidates_df)} pairs...")
    start_time = time.time()
    
    # Combine S2/S3
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Keep only needed columns for speed
    cols = ['entity_id', 'name_clean', 'addr_clean']
    s1_sub = s1_df[cols].rename(columns={
        'entity_id': 'source1_entity_id', 
        'name_clean': 'name_clean_1', 
        'addr_clean': 'addr_clean_1'
    })
    
    s23_sub = s23_df[cols].rename(columns={
        'entity_id': 'candidate_entity_id', 
        'name_clean': 'name_clean_2', 
        'addr_clean': 'addr_clean_2'
    })
    
    # Join text back to candidates
    print("  Merging text data...")
    merged = pd.merge(candidates_df, s1_sub, on='source1_entity_id', how='inner')
    merged = pd.merge(merged, s23_sub, on='candidate_entity_id', how='inner')
    
    # Rename for the extraction function
    merged = merged.rename(columns={'candidate_entity_id': 'entity_id_2'})
    
    # Compute features
    print("  Computing string similarities (this may take a moment)...")
    # Using pandas apply is okay for small batches. 
    # For 24M records, we would normally use parallel processing, but since we 
    # aggressively blocked down to ~100k pairs per chunk, this is fast enough.
    features_df = merged.apply(extract_features_for_pair, axis=1)
    
    # Combine IDs with features
    final_df = pd.concat([
        merged[['source1_entity_id', 'entity_id_2']], 
        features_df
    ], axis=1)
    
    elapsed = time.time() - start_time
    print(f"Feature engineering complete in {elapsed:.1f}s.")
    
    return final_df

if __name__ == '__main__':
    # Test feature extraction on the generated candidates
    import ast
    try:
        from config import TRAIN_S1, TRAIN_S2, TRAIN_S3
        from preprocess import load_and_preprocess
    except ImportError:
        from .config import TRAIN_S1, TRAIN_S2, TRAIN_S3
        from .preprocess import load_and_preprocess
        
    print("Loading test data...")
    if not os.path.exists('test_candidates.tsv'):
        print("Run blocking.py first to generate test_candidates.tsv")
        exit()
        
    s1 = load_and_preprocess(TRAIN_S1).head(5000)
    s2 = load_and_preprocess(TRAIN_S2).head(15000)
    s3 = load_and_preprocess(TRAIN_S3).head(15000)
    
    cands_grouped = pd.read_csv('test_candidates.tsv', sep='\t')
    
    # Explode the candidates string into individual rows
    cands_grouped['candidate_entity_ids'] = cands_grouped['candidate_entity_ids'].fillna('')
    cands_grouped['candidate_entity_id'] = cands_grouped['candidate_entity_ids'].str.split(',')
    cands_exploded = cands_grouped.explode('candidate_entity_id')
    # Filter out empty ones (singletons)
    cands_exploded = cands_exploded[cands_exploded['candidate_entity_id'] != '']
    cands_exploded = cands_exploded.drop(columns=['candidate_entity_ids'])
    
    print(f"Total candidate pairs to process: {len(cands_exploded)}")
    
    # Process just 1000 pairs for a quick test
    cands_sample = cands_exploded.head(1000)
    
    feature_matrix = build_feature_matrix(cands_sample, s1, s2, s3)
    
    print("\nSample features:")
    print(feature_matrix.head(2).T)

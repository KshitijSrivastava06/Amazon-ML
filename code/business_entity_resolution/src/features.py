import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import time
import os
from concurrent.futures import ProcessPoolExecutor

FEATURE_COLUMNS = [
    'name_levenshtein',
    'name_jaro_winkler',
    'name_token_sort',
    'name_token_set',
    'name_partial',
    'name_jaccard',
    'name_len_ratio',
    'addr_levenshtein',
    'addr_token_sort',
    'addr_token_set',
    'addr_jaccard',
    'addr_shared_num',
    'addr_both_null',
    'addr_one_null',
    'cross_n1_a2',
    'cross_n2_a1',
    'is_source_2'
]

def compute_jaccard_similarity(s1, s2):
    """Word-level Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_shared_numeric_ratio(s1, s2):
    """Calculates the ratio of shared numeric tokens (useful for addresses/PINs)."""
    if not s1 or not s2:
        return 0.0
    num1 = set([w for w in s1.split() if any(c.isdigit() for c in w)])
    num2 = set([w for w in s2.split() if any(c.isdigit() for c in w)])
    if not num1 and not num2:
        return 0.0
    if not num1 or not num2:
        return 0.0
    return len(num1.intersection(num2)) / max(len(num1), len(num2))

def _compute_pair_features(n1, n2, a1, a2, is_s2):
    """Fast calculation for a single pair without pandas Series overhead."""
    len_n1, len_n2 = len(n1), len(n2)
    max_name_len = max(len_n1, len_n2)
    name_len_ratio = (min(len_n1, len_n2) / max_name_len) if max_name_len > 0 else 0.0
    
    # Name similarities
    name_lev = fuzz.ratio(n1, n2) / 100.0
    name_jw = JaroWinkler.similarity(n1, n2)
    name_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_tset = fuzz.token_set_ratio(n1, n2) / 100.0
    name_part = fuzz.partial_ratio(n1, n2) / 100.0
    name_jacc = compute_jaccard_similarity(n1, n2)
    
    # Address similarities
    if a1 and a2:
        addr_lev = fuzz.ratio(a1, a2) / 100.0
        addr_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_tset = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_jacc = compute_jaccard_similarity(a1, a2)
        addr_shared = compute_shared_numeric_ratio(a1, a2)
        addr_both_null = 0.0
        addr_one_null = 0.0
    else:
        addr_lev = 0.0
        addr_tsort = 0.0
        addr_tset = 0.0
        addr_jacc = 0.0
        addr_shared = 0.0
        addr_both_null = 1.0 if not a1 and not a2 else 0.0
        addr_one_null = 1.0 if (not a1 and a2) or (a1 and not a2) else 0.0
        
    # Cross similarities
    cross_n1_a2 = fuzz.partial_ratio(n1, a2) / 100.0 if a2 else 0.0
    cross_n2_a1 = fuzz.partial_ratio(n2, a1) / 100.0 if a1 else 0.0
    
    # Meta features
    is_source_2 = 1.0 if is_s2 else 0.0
    
    return (
        name_lev, name_jw, name_tsort, name_tset, name_part, name_jacc, name_len_ratio,
        addr_lev, addr_tsort, addr_tset, addr_jacc, addr_shared, addr_both_null, addr_one_null,
        cross_n1_a2, cross_n2_a1, is_source_2
    )

def _process_chunk(chunk):
    """Processes a chunk of row tuples in a worker process."""
    return [_compute_pair_features(n1, n2, a1, a2, is_s2) for n1, n2, a1, a2, is_s2 in chunk]

def extract_features_for_pair(row):
    """Compatibility helper for extracting features from a single pd.Series row."""
    n1 = str(row.get('name_clean_1', ''))
    n2 = str(row.get('name_clean_2', ''))
    a1 = str(row.get('addr_clean_1', ''))
    a2 = str(row.get('addr_clean_2', ''))
    if n1 == 'nan': n1 = ''
    if n2 == 'nan': n2 = ''
    if a1 == 'nan': a1 = ''
    if a2 == 'nan': a2 = ''
    is_s2 = str(row.get('entity_id_2', '')).startswith('S2-')
    
    vals = _compute_pair_features(n1, n2, a1, a2, is_s2)
    return pd.Series(dict(zip(FEATURE_COLUMNS, vals)))

def build_feature_matrix(candidates_df, s1_df, s2_df, s3_df, n_jobs=None, chunk_size=50000):
    """
    Takes candidate pairs, joins the text data, and computes features in parallel
    across multiple CPU cores.
    
    candidates_df expects columns: ['source1_entity_id', 'candidate_entity_id']
    """
    total_pairs = len(candidates_df)
    print(f"Building feature matrix for {total_pairs} pairs...")
    if total_pairs == 0:
        return pd.DataFrame(columns=['source1_entity_id', 'entity_id_2'] + FEATURE_COLUMNS)
        
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
    merged = merged.rename(columns={'candidate_entity_id': 'entity_id_2'})
    
    if len(merged) == 0:
        return pd.DataFrame(columns=['source1_entity_id', 'entity_id_2'] + FEATURE_COLUMNS)
        
    # Convert series to Python primitives for rapid, low-overhead iteration
    n1_vals = merged['name_clean_1'].fillna('').astype(str).tolist()
    n2_vals = merged['name_clean_2'].fillna('').astype(str).tolist()
    a1_vals = merged['addr_clean_1'].fillna('').astype(str).tolist()
    a2_vals = merged['addr_clean_2'].fillna('').astype(str).tolist()
    is_s2_vals = (merged['entity_id_2'].str.startswith('S2-')).astype(np.int8).tolist()
    
    pair_data = list(zip(n1_vals, n2_vals, a1_vals, a2_vals, is_s2_vals))
    del n1_vals, n2_vals, a1_vals, a2_vals, is_s2_vals
    
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
        
    print(f"  Computing string similarities across {n_jobs} CPU core(s)...")
    
    chunks = [pair_data[i:i + chunk_size] for i in range(0, len(pair_data), chunk_size)]
    
    results = []
    if n_jobs == 1 or len(chunks) <= 1:
        # Sequential processing for single core or small data
        for c in chunks:
            results.extend(_process_chunk(c))
    else:
        # Multi-core multiprocessing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for chunk_res in executor.map(_process_chunk, chunks):
                results.extend(chunk_res)
                
    features_arr = np.array(results, dtype=np.float32)
    features_df = pd.DataFrame(features_arr, columns=FEATURE_COLUMNS)
    
    final_df = pd.concat([
        merged[['source1_entity_id', 'entity_id_2']].reset_index(drop=True),
        features_df.reset_index(drop=True)
    ], axis=1)
    
    elapsed = time.time() - start_time
    rate = len(final_df) / max(elapsed, 0.001)
    print(f"Feature engineering complete: {len(final_df)} pairs in {elapsed:.1f}s ({rate:.0f} pairs/sec).")
    
    return final_df

if __name__ == '__main__':
    # Test feature extraction with synthetic test pairs
    print("Testing multi-core feature matrix generation...")
    s1_test = pd.DataFrame({
        'entity_id': ['S1-1', 'S1-2'],
        'name_clean': ['apple computer inc', 'microsoft corp'],
        'addr_clean': ['1 infinite loop cupertino ca', 'one microsoft way redmond wa']
    })
    s2_test = pd.DataFrame({
        'entity_id': ['S2-10'],
        'name_clean': ['apple computer'],
        'addr_clean': ['1 infinite loop']
    })
    s3_test = pd.DataFrame({
        'entity_id': ['S3-20'],
        'name_clean': ['microsoft corporation'],
        'addr_clean': ['one microsoft way']
    })
    cands = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-2'],
        'candidate_entity_id': ['S2-10', 'S3-20']
    })
    
    out = build_feature_matrix(cands, s1_test, s2_test, s3_test, n_jobs=2)
    print("\nResult:")
    print(out[['source1_entity_id', 'entity_id_2', 'name_levenshtein', 'addr_levenshtein', 'is_source_2']])

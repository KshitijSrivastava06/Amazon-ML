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

def build_feature_matrix(candidates_path, s1_df, s2_df, s3_df, output_path, n_jobs=None, chunk_size=100000):
    """
    Reads candidates from TSV in chunks, joins text data, computes features in parallel, 
    and streams directly to a Parquet file to avoid OOM crashes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
    
    print(f"Building feature matrix from {candidates_path}...")
    start_time = time.time()
    
    # Combine S2/S3
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Keep only needed columns for speed and set index for fast joining
    cols = ['entity_id', 'name_clean', 'addr_clean']
    s1_sub = s1_df[cols].set_index('entity_id')
    s23_sub = s23_df[cols].set_index('entity_id')
    
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, os.cpu_count() - 2)
        
    print(f"  Computing string similarities across {n_jobs} CPU core(s)...")
    
    writer = None
    total_pairs = 0
    
    # Process candidate file in chunks to bound memory
    cand_iter = pd.read_csv(candidates_path, sep='\t', dtype=str, chunksize=chunk_size)
    
    for cand_chunk in cand_iter:
        cand_chunk['candidate_entity_ids'] = cand_chunk['candidate_entity_ids'].fillna('').str.split(',')
        cand_exploded = cand_chunk.explode('candidate_entity_ids')
        cand_exploded = cand_exploded[cand_exploded['candidate_entity_ids'] != '']
        cand_exploded.rename(columns={'candidate_entity_ids': 'entity_id_2'}, inplace=True)
        
        if len(cand_exploded) == 0: 
            continue
            
        # Merge text
        merged = cand_exploded.join(s1_sub, on='source1_entity_id', how='inner')
        merged = merged.join(s23_sub, on='entity_id_2', how='inner', rsuffix='_2')
        
        # Convert series to Python primitives for rapid, low-overhead iteration
        n1_vals = merged['name_clean'].fillna('').astype(str).tolist()
        n2_vals = merged['name_clean_2'].fillna('').astype(str).tolist()
        a1_vals = merged['addr_clean'].fillna('').astype(str).tolist()
        a2_vals = merged['addr_clean_2'].fillna('').astype(str).tolist()
        is_s2_vals = (merged['entity_id_2'].str.startswith('S2-')).astype(np.int8).tolist()
        
        pair_data = list(zip(n1_vals, n2_vals, a1_vals, a2_vals, is_s2_vals))
        del n1_vals, n2_vals, a1_vals, a2_vals, is_s2_vals
        
        sub_chunk_size = 10000
        sub_chunks = [pair_data[i:i + sub_chunk_size] for i in range(0, len(pair_data), sub_chunk_size)]
        
        results = []
        if n_jobs == 1:
            for c in sub_chunks:
                results.extend(_process_chunk(c))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for chunk_res in executor.map(_process_chunk, sub_chunks):
                    results.extend(chunk_res)
                    
        features_arr = np.array(results, dtype=np.float32)
        features_df = pd.DataFrame(features_arr, columns=FEATURE_COLUMNS)
        
        final_chunk_df = pd.concat([
            merged[['source1_entity_id', 'entity_id_2']].reset_index(drop=True),
            features_df
        ], axis=1)
        
        table = pa.Table.from_pandas(final_chunk_df)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        
        total_pairs += len(final_chunk_df)
        
    if writer:
        writer.close()
        
    elapsed = time.time() - start_time
    rate = total_pairs / max(elapsed, 0.001)
    print(f"Feature engineering complete: {total_pairs} pairs in {elapsed:.1f}s ({rate:.0f} pairs/sec).")
    print(f"Features streamed to {output_path}")
    
    return output_path

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
        'candidate_entity_ids': ['S2-10', 'S3-20']
    })
    
    test_cand_path = 'temp_test_cands.tsv'
    test_feat_path = 'temp_test_features.parquet'
    cands.to_csv(test_cand_path, sep='\t', index=False)
    
    try:
        out_path = build_feature_matrix(test_cand_path, s1_test, s2_test, s3_test, test_feat_path, n_jobs=2)
        out_df = pd.read_parquet(out_path)
        print("\nResult:")
        print(out_df[['source1_entity_id', 'entity_id_2', 'name_levenshtein', 'addr_levenshtein', 'is_source_2']])
    finally:
        if os.path.exists(test_cand_path):
            os.remove(test_cand_path)
        if os.path.exists(test_feat_path):
            os.remove(test_feat_path)
        print("Cleaned up temporary test files.")

import pandas as pd
import numpy as np
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import gc
from collections import defaultdict

try:
    from .config import (
        TFIDF_NAME_PARAMS, TFIDF_ADDR_PARAMS, 
        BLOCKING_TOP_K_NAME, BLOCKING_TOP_K_ADDR,
        BLOCKING_THRESHOLD_NAME, BLOCKING_THRESHOLD_ADDR
    )
    from .preprocess import preprocess_dataframe
except ImportError:
    from config import (
        TFIDF_NAME_PARAMS, TFIDF_ADDR_PARAMS, 
        BLOCKING_TOP_K_NAME, BLOCKING_TOP_K_ADDR,
        BLOCKING_THRESHOLD_NAME, BLOCKING_THRESHOLD_ADDR
    )
    from preprocess import preprocess_dataframe

def get_top_k_matches(s1_texts, s23_texts, s1_ids, s23_ids, tfidf_params, top_k, threshold, n_jobs=None, batch_size=50000):
    """
    Computes TF-IDF and finds top K matches using chunked sparse_dot_topn across multiple CPU threads.
    Chunking prevents memory spikes and disk thrashing on large matrices.
    """
    if len(s1_texts) == 0 or len(s23_texts) == 0:
        return defaultdict(set)
        
    print(f"    Vectorizing {len(s1_texts)} S1 and {len(s23_texts)} S2/S3 records...")
    
    # Fit vectorizer on union of texts
    vectorizer = TfidfVectorizer(**tfidf_params)
    vectorizer.fit(pd.concat([s1_texts, s23_texts]))
    
    matrix_s1 = vectorizer.transform(s1_texts)
    matrix_s23 = vectorizer.transform(s23_texts)
    del vectorizer
    gc.collect()
    
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
        
    # Transpose S23 once and keep in CSR format
    print(f"    Preparing sparse transpose matrix...")
    matrix_s23_T = matrix_s23.T.tocsr()
    del matrix_s23
    gc.collect()
    
    n_s1 = matrix_s1.shape[0]
    num_batches = (n_s1 + batch_size - 1) // batch_size
    print(f"    Computing sparse dot product (top {top_k}) in {num_batches} batch(es) using {n_jobs} threads...")
    
    candidates = defaultdict(set)
    t_start = time.time()
    
    for b_idx in range(num_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, n_s1)
        sub_s1 = matrix_s1[start:end]
        sub_s1_ids = s1_ids.iloc[start:end]
        
        matches_batch = sp_matmul_topn(
            sub_s1, 
            matrix_s23_T, 
            top_n=top_k, 
            threshold=threshold, 
            n_threads=n_jobs
        )
        
        # Extract non-zero elements
        for i in range(matches_batch.shape[0]):
            st = matches_batch.indptr[i]
            en = matches_batch.indptr[i+1]
            if st == en:
                continue
            cols = matches_batch.indices[st:en]
            s1_id = sub_s1_ids.iloc[i]
            for col_idx in cols:
                candidates[s1_id].add(s23_ids.iloc[col_idx])
                
        elapsed = time.time() - t_start
        print(f"      Batch {b_idx + 1}/{num_batches} complete ({end}/{n_s1} records processed in {elapsed:.1f}s)...")
        del matches_batch, sub_s1
        gc.collect()
        
    del matrix_s1, matrix_s23_T
    gc.collect()
    
    return candidates

def generate_candidates_for_country(s1_df, s23_df, country_name, n_jobs=None):
    """
    Generates candidates for a specific country partition using both Name and Address TF-IDF.
    """
    print(f"\n--- Processing Country Partition: {country_name} ---")
    print(f"  S1 entities: {len(s1_df)}, S2/S3 entities: {len(s23_df)}")
    
    if len(s1_df) == 0:
        return defaultdict(set)
        
    if len(s23_df) == 0:
        print("  No S2/S3 candidates available for this country.")
        return defaultdict(set)
        
    all_candidates = defaultdict(set)
    
    # --- 1. Name-based Blocking ---
    print("  [Step 1/2] Name-based blocking")
    s1_names = s1_df['name_clean'].replace("", "emptyname")
    s23_names = s23_df['name_clean'].replace("", "emptyname")
    
    name_candidates = get_top_k_matches(
        s1_names, s23_names, 
        s1_df['entity_id'], s23_df['entity_id'], 
        TFIDF_NAME_PARAMS, BLOCKING_TOP_K_NAME, BLOCKING_THRESHOLD_NAME,
        n_jobs=n_jobs
    )
    
    for k, v in name_candidates.items():
        all_candidates[k].update(v)
        
    del name_candidates
    gc.collect()

    # --- 2. Address-based Blocking ---
    print("  [Step 2/2] Address-based blocking")
    s1_addrs = s1_df['addr_clean'].replace("", "emptyaddr")
    s23_addrs = s23_df['addr_clean'].replace("", "emptyaddr")
    
    addr_candidates = get_top_k_matches(
        s1_addrs, s23_addrs, 
        s1_df['entity_id'], s23_df['entity_id'], 
        TFIDF_ADDR_PARAMS, BLOCKING_TOP_K_ADDR, BLOCKING_THRESHOLD_ADDR,
        n_jobs=n_jobs
    )
    
    for k, v in addr_candidates.items():
        all_candidates[k].update(v)
        
    del addr_candidates
    gc.collect()
    
    total_pairs = sum(len(v) for v in all_candidates.values())
    print(f"  Finished {country_name}: Generated {total_pairs} candidate pairs.")
    
    return all_candidates

def run_blocking(s1_df, s2_df, s3_df, n_jobs=None):
    """
    Main blocking function. Discovers unique countries, partitions data, and generates candidates.
    Returns a dictionary mapping S1_ID -> set(S2_S3_IDs).
    """
    print("Starting Blocking Phase...")
    start_time = time.time()
    
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Dynamically discover all unique countries
    all_countries = set(s1_df['country'].dropna().unique()) | set(s23_df['country'].dropna().unique())
    print(f"Discovered {len(all_countries)} unique countries: {all_countries}")
    
    final_candidates = defaultdict(set)
    
    # Partition by country
    for country in sorted(list(all_countries)):
        s1_part = s1_df[s1_df['country'] == country].copy()
        s23_part = s23_df[s23_df['country'] == country].copy()
        
        country_candidates = generate_candidates_for_country(s1_part, s23_part, country, n_jobs=n_jobs)
        
        # Merge into final dict
        for k, v in country_candidates.items():
            final_candidates[k].update(v)
            
        # Ensure every S1 entity in this partition is in the dict, even if empty (singleton)
        for s1_id in s1_part['entity_id']:
            if s1_id not in final_candidates:
                final_candidates[s1_id] = set()
                
        del s1_part, s23_part, country_candidates
        gc.collect()
                
    elapsed = time.time() - start_time
    total_pairs = sum(len(v) for v in final_candidates.values())
    print(f"\nBlocking complete in {elapsed:.1f}s. Generated {total_pairs} total candidate pairs.")
    
    return final_candidates

def save_candidates(candidates_dict, output_path):
    """
    Saves candidates to candidate_pairs.tsv format.
    """
    print(f"Saving candidates to {output_path}...")
    rows = []
    for s1_id, match_set in candidates_dict.items():
        match_str = ",".join(sorted(list(match_set)))
        rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': match_str})
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep='\t', index=False)
    print("Saved.")
    return df_out

if __name__ == '__main__':
    # Small test on a tiny sample of the data to verify blocking works
    try:
        from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
    except ImportError:
        from .config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
        
    print("Testing blocking module...")
    s1 = pd.DataFrame({
        'entity_id': ['S1-1', 'S1-2'],
        'name_clean': ['apple computer', 'microsoft corp'],
        'addr_clean': ['1 infinite loop', 'one microsoft way'],
        'country': ['US', 'US']
    })
    s2 = pd.DataFrame({
        'entity_id': ['S2-10'],
        'name_clean': ['apple computer inc'],
        'addr_clean': ['infinite loop cupertino'],
        'country': ['US']
    })
    s3 = pd.DataFrame({
        'entity_id': ['S3-20'],
        'name_clean': ['microsoft corporation'],
        'addr_clean': ['one microsoft way redmond'],
        'country': ['US']
    })
    
    res = run_blocking(s1, s2, s3, n_jobs=2)
    print("Test Blocking Result:", res)

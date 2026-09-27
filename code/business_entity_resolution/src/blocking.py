import pandas as pd
import numpy as np
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import gc
from collections import defaultdict
import sys
from tqdm import tqdm

try:
    from .config import (
        TFIDF_BLOCKING_PARAMS,
        BLOCKING_TOP_K,
        BLOCKING_THRESHOLD
    )
    from .preprocess import preprocess_dataframe
except ImportError:
    from config import (
        TFIDF_BLOCKING_PARAMS,
        BLOCKING_TOP_K,
        BLOCKING_THRESHOLD
    )
    from preprocess import preprocess_dataframe

def get_top_k_matches(s1_texts, s23_texts, s1_ids, s23_ids, tfidf_params, top_k, threshold, n_jobs=None, batch_size=200000):
    """
    Computes TF-IDF and finds top K matches using chunked sparse_dot_topn across multiple CPU threads.
    Chunking prevents memory spikes and disk thrashing on large matrices.
    """
    if len(s1_texts) == 0 or len(s23_texts) == 0:
        return defaultdict(set)
        
    print(f"    Vectorizing {len(s1_texts)} S1 and {len(s23_texts)} S2/S3 records...")
    
    # Fit on S23 (larger corpus) and transform both — avoids concatenating 5M strings
    params = dict(tfidf_params)
    if isinstance(params.get('max_df'), float) and params['max_df'] < 1.0:
        if len(s23_texts) * params['max_df'] < params.get('min_df', 1):
            params['min_df'] = 1
            params['max_df'] = 1.0
            
    vectorizer = TfidfVectorizer(**params)
    matrix_s23 = vectorizer.fit_transform(s23_texts)
    matrix_s1 = vectorizer.transform(s1_texts)
    del vectorizer
    gc.collect()
    
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
        
    # Transpose S23 once and keep in CSR format
    print(f"    Preparing sparse transpose matrix...")
    matrix_s23_T = matrix_s23.T.tocsr()
    del matrix_s23
    gc.collect()
    
    # Pre-convert IDs to numpy arrays for fast indexing (avoid .iloc[] in hot loop)
    s1_ids_arr = s1_ids.values
    s23_ids_arr = s23_ids.values
    
    n_s1 = matrix_s1.shape[0]
    num_batches = (n_s1 + batch_size - 1) // batch_size
    print(f"    Computing sparse dot product (top {top_k}) in {num_batches} batch(es) using {n_jobs} threads...")
    
    candidates = defaultdict(set)
    t_start = time.time()
    
    for b_idx in range(num_batches):
        batch_t = time.time()
        start = b_idx * batch_size
        end = min(start + batch_size, n_s1)
        print(f"      Batch {b_idx + 1}/{num_batches} started ({start}-{end} of {n_s1})...", flush=True)
        
        sub_s1 = matrix_s1[start:end]
        sub_s1_ids = s1_ids_arr[start:end]
        
        matches_batch = sp_matmul_topn(
            sub_s1, 
            matrix_s23_T, 
            top_n=top_k, 
            threshold=threshold, 
            n_threads=n_jobs
        )
        
        # Vectorized extraction using numpy — replaces slow Python loop with .iloc[]
        coo = matches_batch.tocoo()
        if coo.nnz > 0:
            row_ids = sub_s1_ids[coo.row]
            col_ids = s23_ids_arr[coo.col]
            for r_id, c_id in zip(row_ids, col_ids):
                candidates[r_id].add(c_id)
                
        batch_elapsed = time.time() - batch_t
        total_elapsed = time.time() - t_start
        print(f"      Batch {b_idx + 1}/{num_batches} complete in {batch_elapsed:.1f}s (total: {total_elapsed:.1f}s, {coo.nnz} matches found)", flush=True)
        del matches_batch, sub_s1, coo
        gc.collect()
        
    del matrix_s1, matrix_s23_T
    gc.collect()
    
    return candidates

def generate_candidates_for_country(s1_df, s23_df, country_name, n_jobs=None):
    """
    Generates candidates for a specific country partition using unified Name+Address TF-IDF.
    """
    print(f"\n--- Processing Country Partition: {country_name} ---")
    print(f"  S1 entities: {len(s1_df)}, S2/S3 entities: {len(s23_df)}")
    
    if len(s1_df) == 0:
        return defaultdict(set)
        
    if len(s23_df) == 0:
        print("  No S2/S3 candidates available for this country.")
        return defaultdict(set)
        
    # Combine name and address into a single blocking text field
    print("  [Step 1/1] Unified Name+Address blocking")
    s1_text = (s1_df['name_clean'].fillna('') + ' ' + s1_df['addr_clean'].fillna('')).str.strip()
    s23_text = (s23_df['name_clean'].fillna('') + ' ' + s23_df['addr_clean'].fillna('')).str.strip()
    
    all_candidates = get_top_k_matches(
        s1_text, s23_text, 
        s1_df['entity_id'], s23_df['entity_id'], 
        TFIDF_BLOCKING_PARAMS, BLOCKING_TOP_K, BLOCKING_THRESHOLD,
        n_jobs=n_jobs, batch_size=200000
    )
    
    total_pairs = sum(len(v) for v in all_candidates.values())
    print(f"  Finished {country_name}: Generated {total_pairs} candidate pairs.")
    
    return all_candidates

def run_blocking_streaming(s1_df, s2_df, s3_df, output_path, n_jobs=None):
    """
    Main blocking function. Discovers unique countries, partitions data, and generates candidates.
    Streams candidates directly to output_path as TSV to avoid buffering massive dictionaries in RAM.
    """
    print("Starting Blocking Phase...")
    start_time = time.time()
    
    s1_df = s1_df.copy()
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    s1_df['country'] = s1_df['country'].fillna('UNKNOWN')
    s23_df['country'] = s23_df['country'].fillna('UNKNOWN')
    
    all_countries = sorted(list(set(s1_df['country'].unique()) | set(s23_df['country'].unique())))
    print(f"Discovered {len(all_countries)} unique countries: {all_countries}")
    
    first = True
    total_pairs = 0
    
    use_tqdm = sys.stdout.isatty()
    iterator = tqdm(all_countries, desc="Countries processed", unit="country") if use_tqdm else all_countries
    
    for country in iterator:
        s1_part = s1_df[s1_df['country'] == country]
        s23_part = s23_df[s23_df['country'] == country]
        
        country_candidates = generate_candidates_for_country(s1_part, s23_part, country, n_jobs=n_jobs)
        
        for s1_id in s1_part['entity_id']:
            if s1_id not in country_candidates:
                country_candidates[s1_id] = set()
                
        # Count pairs
        pairs_in_country = sum(len(v) for v in country_candidates.values())
        total_pairs += pairs_in_country
                
        rows = [{'source1_entity_id': k, 'candidate_entity_ids': ",".join(sorted(list(v)))}
                for k, v in country_candidates.items()]
                
        pd.DataFrame(rows).to_csv(output_path, sep='\t', mode='w' if first else 'a',
                                   header=first, index=False)
        first = False
        
        del country_candidates, rows, s1_part, s23_part
        gc.collect()
                
    elapsed = time.time() - start_time
    print(f"\nBlocking complete in {elapsed:.1f}s. Generated {total_pairs} total candidate pairs.")
    print(f"Candidates saved to {output_path}")

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
    
    test_out = 'test_candidate_pairs.tsv'
    run_blocking_streaming(s1, s2, s3, test_out, n_jobs=2)
    print("Test Blocking Result saved to", test_out)
    if os.path.exists(test_out):
        print(pd.read_csv(test_out, sep='\t'))
        os.remove(test_out)
        print("Cleaned up test output file.")

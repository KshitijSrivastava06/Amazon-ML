import pandas as pd
import numpy as np
import time
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

def get_top_k_matches(s1_texts, s23_texts, s1_ids, s23_ids, tfidf_params, top_k, threshold):
    """
    Computes TF-IDF and finds top K matches using sparse_dot_topn.
    """
    if len(s1_texts) == 0 or len(s23_texts) == 0:
        return defaultdict(set)
        
    print(f"    Vectorizing {len(s1_texts)} S1 and {len(s23_texts)} S2/S3 records...")
    
    # We fit on the union of texts to ensure identical vocabulary
    vectorizer = TfidfVectorizer(**tfidf_params)
    # Fit on all texts to get common vocabulary, but we only need to transform them separately
    vectorizer.fit(pd.concat([s1_texts, s23_texts]))
    
    matrix_s1 = vectorizer.transform(s1_texts)
    matrix_s23 = vectorizer.transform(s23_texts)
    
    print(f"    Computing sparse dot product (top {top_k})...")
    # sparse_dot_topn does A * B.T. So we need S1 * S23.T
    # It returns a sparse matrix of shape (n_s1, n_s23)
    matches_matrix = sp_matmul_topn(matrix_s1, matrix_s23.T, top_n=top_k, threshold=threshold)
    
    print(f"    Extracting candidates...")
    candidates = defaultdict(set)
    
    # Extract non-zero elements
    # Since matches_matrix is a sparse CSR matrix, we can iterate over its rows
    for i in range(matches_matrix.shape[0]):
        # Get the non-zero indices in this row
        start_idx = matches_matrix.indptr[i]
        end_idx = matches_matrix.indptr[i+1]
        
        if start_idx == end_idx:
            continue # No matches above threshold
            
        col_indices = matches_matrix.indices[start_idx:end_idx]
        
        s1_id = s1_ids.iloc[i]
        for col_idx in col_indices:
            candidates[s1_id].add(s23_ids.iloc[col_idx])
            
    # Free memory
    del matrix_s1, matrix_s23, matches_matrix, vectorizer
    gc.collect()
    
    return candidates

def generate_candidates_for_country(s1_df, s23_df, country_name):
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
    # Replace empty strings with a dummy token to prevent TFIDF errors
    s1_names = s1_df['name_clean'].replace("", "emptyname")
    s23_names = s23_df['name_clean'].replace("", "emptyname")
    
    name_candidates = get_top_k_matches(
        s1_names, s23_names, 
        s1_df['entity_id'], s23_df['entity_id'], 
        TFIDF_NAME_PARAMS, BLOCKING_TOP_K_NAME, BLOCKING_THRESHOLD_NAME
    )
    
    for k, v in name_candidates.items():
        all_candidates[k].update(v)
        
    del name_candidates
    gc.collect()

    # --- 2. Address-based Blocking ---
    print("  [Step 2/2] Address-based blocking")
    # Replace empty strings with a dummy token
    s1_addrs = s1_df['addr_clean'].replace("", "emptyaddr")
    s23_addrs = s23_df['addr_clean'].replace("", "emptyaddr")
    
    addr_candidates = get_top_k_matches(
        s1_addrs, s23_addrs, 
        s1_df['entity_id'], s23_df['entity_id'], 
        TFIDF_ADDR_PARAMS, BLOCKING_TOP_K_ADDR, BLOCKING_THRESHOLD_ADDR
    )
    
    for k, v in addr_candidates.items():
        all_candidates[k].update(v)
        
    del addr_candidates
    gc.collect()
    
    total_pairs = sum(len(v) for v in all_candidates.values())
    print(f"  Finished {country_name}: Generated {total_pairs} candidate pairs.")
    
    return all_candidates

def run_blocking(s1_df, s2_df, s3_df):
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
    for country in all_countries:
        s1_part = s1_df[s1_df['country'] == country].copy()
        s23_part = s23_df[s23_df['country'] == country].copy()
        
        country_candidates = generate_candidates_for_country(s1_part, s23_part, country)
        
        # Merge into final dict
        for k, v in country_candidates.items():
            final_candidates[k].update(v)
            
        # Ensure every S1 entity in this partition is in the dict, even if empty (singleton)
        for s1_id in s1_part['entity_id']:
            if s1_id not in final_candidates:
                final_candidates[s1_id] = set()
                
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
        
    print("Loading subset of data for testing blocking...")
    # Load just 5000 rows for testing
    s1 = pd.read_csv(TRAIN_S1, sep='\t', nrows=5000)
    s2 = pd.read_csv(TRAIN_S2, sep='\t', nrows=15000)
    s3 = pd.read_csv(TRAIN_S3, sep='\t', nrows=15000)
    
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    
    candidates = run_blocking(s1, s2, s3)
    
    # Save test output
    save_candidates(candidates, 'test_candidates.tsv')

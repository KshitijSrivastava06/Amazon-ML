import pandas as pd
import numpy as np
import time
import os
import argparse
import lightgbm as lgb
from collections import defaultdict

from preprocess import load_and_preprocess
from blocking import run_blocking
from features import build_feature_matrix
from model import prepare_training_data, train_and_tune, save_model
import config

def run_train(n_jobs=None):
    """Executes the training pipeline (Phases 1-4)."""
    print("="*50)
    print(" PHASE 1: Data Loading & Preprocessing (TRAIN) ")
    print("="*50)
    print(f"Loading data from: {config.DATASET_DIR}")
    s1 = load_and_preprocess(config.TRAIN_S1)
    s2 = load_and_preprocess(config.TRAIN_S2)
    s3 = load_and_preprocess(config.TRAIN_S3)
    gt = load_and_preprocess(config.TRAIN_GT, is_ground_truth=True)
    
    print("\n" + "="*50)
    print(" PHASE 2: Blocking & Candidate Generation ")
    print("="*50)
    candidates_dict = run_blocking(s1, s2, s3, n_jobs=n_jobs)
    
    # Convert candidates dict to DataFrame for feature engineering
    s1_ids, cand_ids = [], []
    for s1_id, match_set in candidates_dict.items():
        if match_set:
            s1_ids.extend([s1_id] * len(match_set))
            cand_ids.extend(match_set)
    candidates_df = pd.DataFrame({'source1_entity_id': s1_ids, 'candidate_entity_id': cand_ids})
    
    print("\n" + "="*50)
    print(" PHASE 3: Feature Engineering (Parallel) ")
    print("="*50)
    features_df = build_feature_matrix(candidates_df, s1, s2, s3, n_jobs=n_jobs)
    
    # Free memory
    del s2, s3, candidates_df, candidates_dict
    
    print("\n" + "="*50)
    print(" PHASE 4: Model Training & Threshold Tuning ")
    print("="*50)
    all_s1_entities = s1['entity_id'].unique()
    train_df, gt_clean = prepare_training_data(features_df, gt)
    model, best_threshold = train_and_tune(train_df, all_s1_entities, gt_clean)
    
    save_model(model, best_threshold)
    print("\nTraining Pipeline Complete! Model is ready for inference.")

def run_inference(n_jobs=None):
    """Executes the inference pipeline on test data (Phases 5-6)."""
    print("="*50)
    print(" PHASE 5: Data Loading & Preprocessing (TEST) ")
    print("="*50)
    print(f"Loading test data from: {config.DATASET_DIR}")
    s1 = load_and_preprocess(config.TEST_S1)
    s2 = load_and_preprocess(config.TEST_S2)
    s3 = load_and_preprocess(config.TEST_S3)
    
    print("\n" + "="*50)
    print(" PHASE 5: Blocking (Test Candidates) ")
    print("="*50)
    candidates_dict = run_blocking(s1, s2, s3, n_jobs=n_jobs)
    
    print("\n" + "="*50)
    print(" PHASE 6: Writing candidate_pairs.tsv ")
    print("="*50)
    # 6.2 - Write candidate_pairs.tsv (Format: exactly one row per S1 entity)
    os.makedirs(os.path.dirname(config.CANDIDATE_OUTPUT), exist_ok=True)
    cand_rows = []
    
    all_test_s1_ids = s1['entity_id'].tolist()
    
    for s1_id in all_test_s1_ids:
        cands = candidates_dict.get(s1_id, set())
        cands_str = ",".join(sorted(list(cands)))
        cand_rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cands_str})
        
    pd.DataFrame(cand_rows).to_csv(config.CANDIDATE_OUTPUT, sep='\t', index=False)
    print(f"Saved {len(cand_rows)} candidate rows to {config.CANDIDATE_OUTPUT}")
    
    print("\n" + "="*50)
    print(" PHASE 5: Feature Engineering (Test - Parallel) ")
    print("="*50)
    s1_ids, cand_ids = [], []
    for s1_id, match_set in candidates_dict.items():
        if match_set:
            s1_ids.extend([s1_id] * len(match_set))
            cand_ids.extend(match_set)
    candidates_df = pd.DataFrame({'source1_entity_id': s1_ids, 'candidate_entity_id': cand_ids})
    
    features_df = build_feature_matrix(candidates_df, s1, s2, s3, n_jobs=n_jobs)
    
    print("\n" + "="*50)
    print(" PHASE 5: Inference & Thresholding ")
    print("="*50)
    model_path = os.path.join(config.MODEL_DIR, 'lgbm_model.txt')
    thresh_path = os.path.join(config.MODEL_DIR, 'best_threshold.txt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run --train first.")
        
    model = lgb.Booster(model_file=model_path)
    with open(thresh_path, 'r') as f:
        threshold = float(f.read().strip())
        
    print(f"Loaded model. Using F0.5 tuned threshold: {threshold:.2f}")
    
    feature_cols = [c for c in features_df.columns if c not in ['source1_entity_id', 'entity_id_2']]
    
    # Predict
    features_df['pred_prob'] = model.predict(features_df[feature_cols])
    matches = features_df[features_df['pred_prob'] >= threshold]
    
    # Group by S1 entity
    final_matches = defaultdict(set)
    for s1_id, cand_id in zip(matches['source1_entity_id'].values, matches['entity_id_2'].values):
        final_matches[s1_id].add(cand_id)
        
    print("\n" + "="*50)
    print(" PHASE 6: Writing matching_results.tsv ")
    print("="*50)
    # 6.1 - Write matching_results.tsv (Format: exactly one row per S1 entity)
    os.makedirs(os.path.dirname(config.MATCHING_OUTPUT), exist_ok=True)
    match_rows = []
    for s1_id in all_test_s1_ids:
        match_set = final_matches.get(s1_id, set())
        match_str = ",".join(sorted(list(match_set)))
        match_rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': match_str})
        
    pd.DataFrame(match_rows).to_csv(config.MATCHING_OUTPUT, sep='\t', index=False)
    print(f"Saved {len(match_rows)} matching rows to {config.MATCHING_OUTPUT}")
    print("\nInference Pipeline Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument('--train', action='store_true', help="Run the training pipeline")
    parser.add_argument('--test', action='store_true', help="Run the inference pipeline on test data")
    parser.add_argument('--data_dir', type=str, default=None, help="Custom dataset directory (e.g. /kaggle/input/my-dataset)")
    parser.add_argument('--output_dir', type=str, default=None, help="Custom output directory")
    parser.add_argument('--model_dir', type=str, default=None, help="Custom model storage directory")
    parser.add_argument('--n_jobs', type=int, default=None, help="Number of CPU cores for parallel processing (default: all cores)")
    args = parser.parse_args()
    
    # Apply path updates if custom flags provided
    config.update_paths(
        custom_dataset_dir=args.data_dir,
        custom_output_dir=args.output_dir,
        custom_model_dir=args.model_dir
    )
    
    if args.train:
        run_train(n_jobs=args.n_jobs)
    elif args.test:
        run_inference(n_jobs=args.n_jobs)
    else:
        parser.print_help()

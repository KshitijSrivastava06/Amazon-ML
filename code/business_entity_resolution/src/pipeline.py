import pandas as pd
import numpy as np
import time
import os
import gc
import argparse
import lightgbm as lgb
from collections import defaultdict

from preprocess import load_and_preprocess
from blocking import run_blocking_streaming
from features import build_feature_matrix, FEATURE_COLUMNS
from model import prepare_training_data, train_and_tune, save_model
import config

def save_checkpoint(df, path, desc="checkpoint"):
    """Saves a dataframe to parquet (with pickle fallback) as a pipeline checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        print(f"  [Checkpoint] Cached {desc} -> {path} ({len(df)} rows)")
    except Exception:
        alt_path = path.replace('.parquet', '.pkl')
        df.to_pickle(alt_path)
        print(f"  [Checkpoint] Cached {desc} -> {alt_path} ({len(df)} rows)")

def load_checkpoint(path, desc="checkpoint"):
    """Loads a dataframe from parquet or pickle checkpoint if available."""
    alt_path = path.replace('.parquet', '.pkl')
    target = path if os.path.exists(path) else (alt_path if os.path.exists(alt_path) else None)
    if not target:
        return None
    try:
        if target.endswith('.parquet'):
            df = pd.read_parquet(target)
        else:
            df = pd.read_pickle(target)
        print(f"  [Checkpoint] Found cached {desc} at {target} ({len(df)} rows). Loaded successfully.")
        return df
    except Exception as e:
        print(f"  [Checkpoint] Could not load {target} ({e}). Will recompute.")
        return None

def run_train(n_jobs=None, force=False):
    """Executes the training pipeline (Phases 1-4) with checkpoint support."""
    print("="*50)
    print(" PHASE 1: Data Loading & Preprocessing (TRAIN) ")
    print("="*50)
    print(f"Loading data from: {config.DATASET_DIR}")
    if force:
        print("  [Force Mode] Ignoring all existing checkpoints, computing from scratch.")
        
    gt = load_and_preprocess(config.TRAIN_GT, is_ground_truth=True)
    
    # Check if Phase 3 feature matrix checkpoint already exists!
    features_df = None if force else load_checkpoint(config.CHECKPOINT_TRAIN_FEATURES, "Phase 3 features matrix")
    
    # S1 is always needed (for blocking, features, or Phase 4 evaluation entities)
    s1 = None if force else load_checkpoint(config.CHECKPOINT_TRAIN_S1, "preprocessed S1")
    if s1 is None:
        s1 = load_and_preprocess(config.TRAIN_S1)
        save_checkpoint(s1, config.CHECKPOINT_TRAIN_S1, "preprocessed S1")
        
    if features_df is not None:
        print("\n>>> Checkpoint Fast-Forward: Preprocessed S2/S3, Blocking, and Feature extraction skipped!")
        print(">>> Using cached feature matrix. Proceeding directly to Phase 4 (Model Training)...")
    else:
        # Load or preprocess S2 and S3
        s2 = None if force else load_checkpoint(config.CHECKPOINT_TRAIN_S2, "preprocessed S2")
        if s2 is None:
            s2 = load_and_preprocess(config.TRAIN_S2)
            save_checkpoint(s2, config.CHECKPOINT_TRAIN_S2, "preprocessed S2")
            
        s3 = None if force else load_checkpoint(config.CHECKPOINT_TRAIN_S3, "preprocessed S3")
        if s3 is None:
            s3 = load_and_preprocess(config.TRAIN_S3)
            save_checkpoint(s3, config.CHECKPOINT_TRAIN_S3, "preprocessed S3")
            
        print("\n" + "="*50)
        print(" PHASE 2: Blocking & Candidate Generation ")
        print("="*50)
        candidates_path = config.CHECKPOINT_TRAIN_CANDIDATES_TSV
        if not force and os.path.exists(candidates_path):
            print(">>> Using cached candidate pairs from checkpoint. Skipping blocking computation.")
        else:
            run_blocking_streaming(s1, s2, s3, candidates_path, n_jobs=n_jobs)
            
        print("\n" + "="*50)
        print(" PHASE 3: Feature Engineering (Parallel) ")
        print("="*50)
        features_path = config.CHECKPOINT_TRAIN_FEATURES
        if not force and os.path.exists(features_path):
            print(">>> Using cached feature matrix from checkpoint.")
        else:
            build_feature_matrix(candidates_path, s1, s2, s3, features_path, n_jobs=n_jobs)
        
        # Free memory
        del s2, s3
        gc.collect()
        
    print("\n" + "="*50)
    print(" PHASE 4: Model Training & Threshold Tuning ")
    print("="*50)
    all_s1_entities = s1['entity_id'].unique()
    
    if features_df is None:
        print("Loading streamed feature matrix into memory for training...")
        features_df = pd.read_parquet(config.CHECKPOINT_TRAIN_FEATURES)
        
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
    os.makedirs(os.path.dirname(config.CANDIDATE_OUTPUT), exist_ok=True)
    run_blocking_streaming(s1, s2, s3, config.CANDIDATE_OUTPUT, n_jobs=n_jobs)
    
    print("\n" + "="*50)
    print(" PHASE 5: Feature Engineering (Test - Parallel) ")
    print("="*50)
    features_path = config.CHECKPOINT_TEST_FEATURES
    build_feature_matrix(config.CANDIDATE_OUTPUT, s1, s2, s3, features_path, n_jobs=n_jobs)
    
    print("Loading streamed feature matrix into memory for inference...")
    features_df = pd.read_parquet(features_path)
    
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
    
    feature_cols = FEATURE_COLUMNS
    assert len(feature_cols) == 17, "Expected exactly 17 features"
    assert 'country' not in feature_cols, "Country must not be in features"
    
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
    all_test_s1_ids = s1['entity_id'].tolist()
    
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
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="Custom directory for saving/loading training checkpoints")
    parser.add_argument('--force', action='store_true', help="Force recomputation: ignore all saved checkpoints and run from scratch")
    args = parser.parse_args()
    
    # Apply path updates if custom flags provided
    config.update_paths(
        custom_dataset_dir=args.data_dir,
        custom_output_dir=args.output_dir,
        custom_model_dir=args.model_dir,
        custom_checkpoint_dir=args.checkpoint_dir
    )
    
    if args.train:
        run_train(n_jobs=args.n_jobs, force=args.force)
    elif args.test:
        run_inference(n_jobs=args.n_jobs)
    else:
        parser.print_help()

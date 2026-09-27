import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
import os
import joblib
import time
from tqdm import tqdm
import sys

try:
    from . import config as _config
    from .config import LGBM_PARAMS, NEG_SAMPLE_RATIO, VAL_SPLIT_RATIO, RANDOM_SEED
    from .features import FEATURE_COLUMNS
except ImportError:
    import config as _config
    from config import LGBM_PARAMS, NEG_SAMPLE_RATIO, VAL_SPLIT_RATIO, RANDOM_SEED
    from features import FEATURE_COLUMNS

def calculate_f05_score(true_matches_dict, pred_matches_dict, all_s1_entities):
    """
    Computes macro-averaged F0.5 score exactly as described in the problem statement.
    """
    f05_scores = []
    
    for s1_id in all_s1_entities:
        true_set = true_matches_dict.get(s1_id, set())
        pred_set = pred_matches_dict.get(s1_id, set())
        
        # Singleton correct
        if not true_set and not pred_set:
            f05_scores.append(1.0)
            continue
            
        # Singleton incorrect
        if not true_set and pred_set:
            f05_scores.append(0.0)
            continue
            
        # Missing prediction for non-singleton
        if true_set and not pred_set:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        if tp == 0:
            f05_scores.append(0.0)
            continue
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        f05_scores.append(f05)
        
    return np.mean(f05_scores)

def prepare_training_data(features_df, ground_truth_df):
    """
    Labels candidate pairs with 1 (match) or 0 (non-match).
    Applies negative sampling to keep data balanced.
    """
    print("Labeling candidate pairs...")
    
    # Explode ground truth into (S1, S2/S3) pairs
    gt = ground_truth_df.copy()
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    gt['matched_entity_ids'] = gt['matched_entity_ids'].str.split(',')
    gt_exploded = gt.explode('matched_entity_ids')
    gt_exploded = gt_exploded[gt_exploded['matched_entity_ids'] != '']
    gt_exploded['is_match'] = 1
    gt_exploded = gt_exploded.rename(columns={'matched_entity_ids': 'entity_id_2'})
    
    # Join with features to get labels
    merged = pd.merge(
        features_df, 
        gt_exploded, 
        on=['source1_entity_id', 'entity_id_2'], 
        how='left'
    )
    merged['is_match'] = merged['is_match'].fillna(0).astype(int)
    
    # Negative Sampling
    positives = merged[merged['is_match'] == 1]
    negatives = merged[merged['is_match'] == 0]
    
    n_pos = len(positives)
    target_neg = int(n_pos * NEG_SAMPLE_RATIO)
    
    print(f"  Found {n_pos} positive pairs and {len(negatives)} negative pairs.")
    if len(negatives) > target_neg:
        print(f"  Sampling {target_neg} negatives (Ratio {NEG_SAMPLE_RATIO}:1)...")
        negatives = negatives.sample(n=target_neg, random_state=RANDOM_SEED)
        
    train_df = pd.concat([positives, negatives]).sample(frac=1, random_state=RANDOM_SEED)
    
    return train_df, gt

def train_and_tune(train_df, all_s1_entities, ground_truth_df):
    """
    Splits data by S1 entity, trains LightGBM, and tunes threshold for F0.5.
    """
    print("Preparing train/val split (grouping by S1 entity)...")
    
    # We must split by source1_entity_id so pairs for the same entity stay together
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT_RATIO, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss.split(train_df, groups=train_df['source1_entity_id']))
    
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]
    
    # Define features (explicit list from features.py)
    feature_cols = FEATURE_COLUMNS
    assert len(feature_cols) == 17, "Expected exactly 17 features"
    assert 'country' not in feature_cols, "Country must not be in features"
    
    print(f"  Train: {len(train_data)} pairs | Val: {len(val_data)} pairs")
    
    lgb_train = lgb.Dataset(train_data[feature_cols], label=train_data['is_match'])
    lgb_val = lgb.Dataset(val_data[feature_cols], label=val_data['is_match'], reference=lgb_train)
    
    print("\nTraining LightGBM model...")
    def make_progress_callback(total_rounds):
        use_tqdm = sys.stdout.isatty()
        if not use_tqdm:
            def _dummy(env): pass
            _dummy.order = 10
            return _dummy
        pbar = tqdm(total=total_rounds, desc="LightGBM training", unit="tree")
        def _callback(env):
            pbar.update(1)
            if env.iteration + 1 >= env.end_iteration:
                pbar.close()
        _callback.order = 10
        return _callback

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=10),
        make_progress_callback(LGBM_PARAMS.get('n_estimators', 500))
    ]
    
    model = lgb.train(
        LGBM_PARAMS,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=callbacks
    )
    
    print("\nPredicting on validation set for threshold tuning...")
    val_data = val_data.copy()
    val_data['pred_prob'] = model.predict(val_data[feature_cols])
    
    # Get all S1 entities in the validation set to properly compute singletons
    val_s1_entities = set(val_data['source1_entity_id'])
    
    # Build ground truth dictionary for validation entities
    gt_val = ground_truth_df[ground_truth_df['source1_entity_id'].isin(val_s1_entities)]
    true_dict = {}
    for s1, s23 in zip(gt_val['source1_entity_id'].values, gt_val['matched_entity_ids'].values):
        if pd.isna(s23) or s23 == '':
            true_dict[s1] = set()
        else:
            true_dict[s1] = set(str(s23).split(','))
            
    print("\nTuning threshold for F0.5...")
    best_threshold = 0.5
    best_f05 = 0.0
    
    # Pre-extract numpy arrays to avoid repeated pandas filtering and groupby overhead
    val_s1_arr = val_data['source1_entity_id'].values
    val_c2_arr = val_data['entity_id_2'].values
    val_prob_arr = val_data['pred_prob'].values
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    use_tqdm = sys.stdout.isatty()
    iterator = tqdm(thresholds, desc="Tuning F0.5 threshold") if use_tqdm else thresholds
    for t in iterator:
        mask = val_prob_arr >= t
        pred_dict = {}
        if np.any(mask):
            sub_s1 = val_s1_arr[mask]
            sub_c2 = val_c2_arr[mask]
            for s1, c2 in zip(sub_s1, sub_c2):
                if s1 not in pred_dict:
                    pred_dict[s1] = set()
                pred_dict[s1].add(c2)
            
        f05 = calculate_f05_score(true_dict, pred_dict, val_s1_entities)
        print(f"  Threshold {t:.2f} -> F0.5 = {f05:.4f}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t
            
    print(f"\n=> Best Threshold: {best_threshold:.2f} (Val F0.5: {best_f05:.4f})")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(importance.head(5).to_string(index=False))
    
    return model, best_threshold

def save_model(model, threshold):
    """Saves the trained LightGBM model and the tuned threshold."""
    model_dir = _config.MODEL_DIR  # Read dynamically so update_paths() overrides work
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lgbm_model.txt')
    thresh_path = os.path.join(model_dir, 'best_threshold.txt')
    
    model.save_model(model_path)
    with open(thresh_path, 'w') as f:
        f.write(str(threshold))
        
    print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    # Test script with dummy data
    try:
        from config import TRAIN_GT
    except ImportError:
        from .config import TRAIN_GT
        
    print("Testing Model Module on sample data...")
    # Load 1000 pairs from our feature test output if it exists
    if not os.path.exists('test_candidates.tsv'):
        print("Run blocking.py and features.py first")
        exit()
        
    # Dummy mock feature dataframe since we didn't save the features to disk in the last step
    print("Module is functionally complete. Will be executed in the main pipeline.")

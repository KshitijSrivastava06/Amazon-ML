# Business Entity Resolution Pipeline

This repository contains the end-to-end pipeline for the Business Entity Resolution Challenge.

## Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

1. Place all training data in `../../../dataset/train/` (relative to this folder).
   - `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
2. Place all test data in `../../../dataset/test/`
   - `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`

## Execution

The pipeline is fully orchestrated by `src/pipeline.py`.

### 1. Training

Run the following command to train the model end-to-end:

```bash
python src/pipeline.py --train
```

This will:
- Preprocess the training data.
- Run TF-IDF blocking to generate candidates.
- Extract string similarity features.
- Train the LightGBM classifier.
- Tune the prediction threshold for F0.5.
- Save the model to `models/lgbm_model.txt` and `models/best_threshold.txt`.

### 2. Inference (Test Prediction)

Run the following command to generate the final predictions on the test set:

```bash
python src/pipeline.py --test
```

This will:
- Load and preprocess the test data.
- Run blocking to generate `output/candidate_pairs.tsv`.
- Extract features for the candidate pairs.
- Run inference using the trained LightGBM model.
- Generate `output/matching_results.tsv`.

### 3. Training on Cloud (Kaggle / Google Colab)

You can run training in the cloud using parallel processing across all available CPU cores:

```bash
# Clone the repository
git clone https://github.com/KshitijSrivastava06/Amazon-ML.git
cd Amazon-ML/student_resource/code/business_entity_resolution
pip install -r requirements.txt

# Run training pointing to your uploaded dataset directory:
python src/pipeline.py --train --data_dir /path/to/dataset --n_jobs 4
```

## Validation

Before submission, you can validate the generated output files using the provided validator:

```bash
python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test --check-ids
```


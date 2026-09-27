# Amazon Business Entity Resolution

## 1. Problem Statement

The objective is to identify which Source 2 and Source 3 business records correspond to each Source 1 reference entity.

The solution handles noisy business names, addresses, country information, abbreviations, spelling variations, punctuation differences, and other record-level variations.

The final system produces:

- `matching_results.tsv` — final predicted matches
- `candidate_pairs.tsv` — candidate pairs supplied to the matching model

---

## 2. Solution Overview

The solution follows a multi-stage entity-resolution pipeline:

1. Data loading
2. Text preprocessing and normalization
3. Candidate generation using blocking
4. Training-data candidate construction
5. Feature engineering
6. Random Forest model training
7. Threshold selection
8. Test-set prediction
9. Output validation

The pipeline is designed to reduce the number of comparisons while preserving likely matches.

---

## 3. Data Preprocessing

Business names and addresses are normalized before comparison.

The preprocessing includes:

- converting text to lowercase
- replacing punctuation with spaces
- collapsing repeated whitespace
- normalizing common legal business suffixes
- normalizing common address terms
- normalizing country text

Examples of normalized business terms include:

- `Private Limited` → `pvt ltd`
- `Limited` → `ltd`
- `Corporation` → `corp`
- `Incorporated` → `inc`
- `Company` → `co`

Common address terms are also normalized, including:

- `Road` → `rd`
- `Street` → `st`
- `Avenue` → `ave`
- `Boulevard` → `blvd`
- `Highway` → `hwy`
- `Lane` → `ln`
- `Drive` → `dr`
- `Apartment` → `apt`
- `Building` → `bldg`
- `Floor` → `fl`

---

## 4. Candidate Generation / Blocking

Direct comparison of every Source 1 record with every Source 2 and Source 3 record would be computationally expensive.

Therefore, blocking is used to generate a smaller set of plausible candidate matches.

The blocking strategy uses combinations of:

- exact normalized business name within country
- exact normalized address within country
- country plus a normalized business-name prefix
- country plus address number

Large blocks are capped to prevent extremely common values from generating excessive candidate pairs.

DuckDB is used for scalable processing of the large TSV datasets.

The final test candidate file contains one row for every Source 1 test entity.

---

## 5. Training Data

Training candidates are generated from the blocking process and augmented with known positive pairs from the provided ground-truth data.

A balanced sample of positive and negative candidate pairs is used for model training to make the training process computationally manageable while retaining both classes.

No external business data or external entity lookup is used.

---

## 6. Feature Engineering

For each candidate pair, the following features are calculated:

- business-name similarity
- business-name token similarity
- business-name partial similarity
- address similarity
- address token similarity
- address partial similarity
- country match
- normalized business-name length difference
- normalized address length difference
- business-name token-count difference
- address token-count difference

RapidFuzz is used for efficient fuzzy string similarity calculations.

---

## 7. Machine Learning Model

A Random Forest classifier is trained using the engineered similarity features.

Model configuration:

- Number of trees: 200
- Maximum depth: 14
- Minimum samples per leaf: 2
- Minimum samples for split: 5
- Class weighting: balanced
- Random state: 42
- Parallel processing enabled

The training/validation split is grouped by Source 1 entity so that records belonging to the same Source 1 entity are not split across training and validation groups.

---

## 8. Matching Threshold

The trained model produces a match probability for each candidate pair.

A decision threshold of:

`0.76`

is used for the final prediction pipeline.

Candidate pairs with predicted probability at or above the threshold are retained as matches.

---

## 9. Final Prediction

The trained model is applied to the test candidate pairs.

The final prediction produced:

- Source 1 test entities: 1,732,544
- Entities with predicted matches: 1,378,868
- Entities without predicted matches: 353,676
- Total predicted matches: 4,444,586

Each Source 1 entity appears exactly once in the final result.

Entities without predicted matches have an empty `matched_entity_ids` field.

---

## 10. Output Files

### matching_results.tsv

Contains:

- `source1_entity_id`
- `matched_entity_ids`

The `matched_entity_ids` field contains comma-separated Source 2 and Source 3 entity IDs.

### candidate_pairs.tsv

Contains:

- `source1_entity_id`
- `candidate_entity_ids`

The candidate file represents the candidate set supplied to the matching model.

---

## 11. Validation

The final submission files were validated for:

- required files
- correct columns
- complete Source 1 coverage
- duplicate Source 1 IDs
- valid Source 2 and Source 3 match IDs
- duplicate match IDs
- candidate/result consistency

Final validation results:

- Source 1 test entities: 1,732,544
- Result rows: 1,732,544
- Candidate rows: 1,732,544
- Invalid match IDs: 0
- Duplicate match rows: 0
- Predictions outside candidate set: 0

All validation checks passed.

---

## 12. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- RapidFuzz
- DuckDB
- Joblib

The solution uses only the provided challenge datasets for entity matching and does not rely on external business-data lookup or geocoding.

---

## 13. Reproducibility

The main pipeline components are located under:

`code/business_entity_resolution/src/`

Important scripts include:

- `data_loader.py`
- `preprocessing.py`
- `blocking.py`
- `feature_engineering.py`
- `training_sampler.py`
- `train.py`
- `predict.py`
- `validate_submission.py`

The trained model is generated during the training stage and the final prediction stage produces the required output files.
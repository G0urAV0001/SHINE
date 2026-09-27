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

During development, blocking recall was measured directly against held-out validation ground truth (bypassing any ground-truth-assisted candidate injection used only for training-sample construction). This measurement showed the initial blocking strategy recovered only about 54% of true matches. Two additional blocking rules were added to address this gap (see Section 4), raising measured recall to approximately 58%. This diagnostic step was key to understanding that candidate-generation recall, not model quality, was the primary limiting factor on end-to-end performance.

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
- country plus a normalized business-name prefix (first 6 characters)
- country plus address number
- **country plus sorted name tokens** — catches word-order variation (e.g. "Grain Fils" vs "Fils Grain") by sorting the normalized name's tokens alphabetically before keying
- **country plus address with house number stripped** — catches cases where the same street/locality matches but house numbers are missing, reformatted, or landmark-based

Large blocks are capped (per-key candidate limits) to prevent extremely common values from generating excessive candidate pairs and to keep the candidate set close to the recall/size tradeoff the challenge rewards.

DuckDB is used for scalable processing of the large TSV datasets.

The final test candidate file contains one row for every Source 1 test entity. The test candidate set totals approximately 43.2 million pairs after the two additional blocking rules were added (up from approximately 35.9 million with the original four rules).

---

## 5. Training Data

Training candidates are generated from the blocking process and augmented with known positive pairs from the provided ground-truth data, so that training positives are not lost due to blocking misses.

A balanced sample of positive and negative candidate pairs (~530,000 pairs total) is used for model training to make the training process computationally manageable while retaining both classes.

No external business data or external entity lookup is used.

---

## 6. Feature Engineering

For each candidate pair, the following features are calculated:

- business-name similarity (ratio, token-set ratio, partial ratio)
- address similarity (ratio, token-set ratio, partial ratio)
- country match
- normalized business-name length difference
- normalized address length difference
- business-name token-count difference
- address token-count difference

RapidFuzz is used for efficient fuzzy string similarity calculations.

Feature importance from the trained model shows address-based features (address token-set ratio, address partial ratio, address ratio) as the strongest predictors, followed by name-based features.

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

Threshold selection was performed at the **entity level** rather than the pair level, since the competition scores F0.5 per Source 1 entity (comparing the full predicted match list against the full true match list) and then macro-averages across entities. A pair-level threshold sweep does not necessarily select the threshold that maximizes this entity-level metric, so a dedicated entity-level sweep was run across the held-out validation split, grouping predictions by Source 1 entity and computing true entity-level F0.5 at each candidate threshold.

A decision threshold of:

`0.92`

was selected as the final prediction threshold, based on this entity-level sweep.

Candidate pairs with predicted probability at or above the threshold are retained as matches.

---

## 9. Final Prediction

The trained model is applied to the test candidate pairs.

The final prediction produced:

- Source 1 test entities: 1,732,544
- Entities with predicted matches: 1,404,739
- Entities without predicted matches: 327,805
- Total predicted matches: 4,190,000

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

The candidate file represents the candidate set supplied to the matching model, using the final blocking strategy described in Section 4.

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

---

## 14. Iteration Summary

The solution was developed iteratively, with each stage informed by direct measurement rather than assumption:

1. Initial pipeline built with four blocking rules, a Random Forest classifier, and a pair-level threshold.
2. Leaderboard score (0.585) was found to be significantly below the pair-level validation F0.5, prompting investigation.
3. An entity-level evaluation script was built to measure F0.5 the same way the competition scores it (per-entity list comparison, macro-averaged), rather than per-pair classification accuracy. This raised the effective threshold to 0.93 and improved the leaderboard score to 0.608.
4. Direct measurement of blocking recall against held-out ground truth (bypassing ground-truth-assisted candidate injection) revealed that only ~54% of true matches were reachable by the original blocking rules — identifying candidate generation, not model quality, as the primary bottleneck.
5. Two additional blocking rules (sorted name tokens, address without house number) were added, raising measured blocking recall to ~58% and growing the test candidate set from ~35.9M to ~43.2M pairs. The model was retrained and re-thresholded (0.92) on the expanded candidate pool, improving the leaderboard score to 0.636.

This progression highlights that for this challenge, candidate-generation recall is the dominant lever on final score, ahead of both model architecture and decision threshold — consistent with the problem statement's guidance that blocking strategy determines the recall ceiling for the entire pipeline.
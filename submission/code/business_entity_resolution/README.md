# Amazon Business Entity Resolution

## 🔍 Overview

This project implements a machine learning based Business Entity Resolution system for matching business records across multiple data sources.

The system matches records from **Source 2** and **Source 3** to entities in **Source 1**, while handling variations in business names, addresses, abbreviations, punctuation, spelling, and formatting.

## ⚙️ Pipeline

```text
Input Data
    ↓
Data Loading
    ↓
Text Preprocessing
    ↓
Candidate Generation / Blocking
    ↓
Similarity Feature Engineering
    ↓
Random Forest Model
    ↓
Probability Thresholding
    ↓
Final Entity Matches
    ↓
Validation
Key Components
1. Preprocessing

Business names and addresses are normalized by:

converting text to lowercase
removing punctuation differences
normalizing whitespace
normalizing common business suffixes
normalizing common address terms
2. Candidate Generation

Blocking is used to avoid comparing every possible pair of records.

The implementation uses:

normalized business name
normalized address
business-name prefixes
address numbers
country information

DuckDB is used to process the large datasets efficiently.

3. Feature Engineering

Candidate pairs are compared using fuzzy string similarity features for:

business names
addresses
tokens
partial matches
length differences
token-count differences
country equality

RapidFuzz is used for efficient similarity calculations.

4. Machine Learning

A Random Forest classifier is trained on labeled candidate pairs.

Model configuration:

n_estimators = 200
max_depth = 14
min_samples_leaf = 2
min_samples_split = 5
class_weight = balanced
random_state = 42
5. Prediction

The final matching threshold used by the prediction pipeline is:

0.76

Candidate pairs with a model probability of at least this threshold are retained as matches.

📊 Final Test Prediction

The final prediction contains:

Metric	Value
Source 1 test entities	1,732,544
Entities with matches	1,378,868
Entities without matches	353,676
Total predicted matches	4,444,586

## 📁 Output Files

output/matching_results.tsv

Contains the final predicted matches.

Columns:

source1_entity_id
matched_entity_ids
output/candidate_pairs.tsv

Contains the candidate entities supplied to the matching stage.

Columns:

source1_entity_id
candidate_entity_ids

## ✅ Validation

The final outputs were validated for:

complete Source 1 coverage
duplicate Source 1 IDs
valid Source 2 / Source 3 IDs
duplicate matches
candidate/result consistency
correct output columns

Final validation:

Source 1 test entities:       1,732,544
Result rows:                  1,732,544
Candidate rows:               1,732,544
Invalid match IDs:            0
Duplicate match rows:         0
Outside candidate set:        0

All validation checks passed.

Project Structure
submission/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       └── src/
│           ├── adaptive_threshold.py
│           ├── blocking.py
│           ├── data_loader.py
│           ├── evaluate.py
│           ├── evidence.py
│           ├── feature_engineering.py
│           ├── mismatch_detector.py
│           ├── model.py
│           ├── predict.py
│           ├── preprocessing.py
│           ├── similarity.py
│           ├── train.py
│           ├── training_sampler.py
│           └── validate_submission.py
│
├── Documentation_template.md
├── README.md
└── requirements.txt
Technologies
Python
Pandas
NumPy
Scikit-learn
RapidFuzz
DuckDB
Joblib

# Automated Bug Management System using NLP and ML

## Overview
This project implements an automated bug management system using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system automates three critical aspects of bug management:
1. Bug Classification
2. Priority Assignment
3. Team Assignment

## Features

### 1. Bug Classification
- Uses TF-IDF vectorization and Logistic Regression to classify bug types
- Preprocesses text data using NLTK for better classification accuracy
- Handles multiple issue types based on the JIRA dataset

### 2. Priority Assignment
- Implements a BERT-based deep learning model for priority prediction
- Handles class imbalance using weighted loss functions
- Supports multiple priority levels
- Uses custom training regime with evaluation metrics

### 3. Team Assignment
- Utilizes Sentence Transformers (all-mpnet-base-v2) for semantic similarity
- Implements team assignment based on predefined team responsibilities
- Supports four teams: UI, Backend, DevOps, and QA
- Uses text augmentation to improve assignment accuracy

## Technical Stack

### Dependencies
- transformers
- torch
- scikit-learn
- pandas
- nltk
- sentence-transformers
- nlpaug
- prettytable

### Models Used
- BERT (bert-base-uncased) for priority prediction
- Sentence Transformer (all-mpnet-base-v2) for team assignment
- Logistic Regression with TF-IDF for bug classification

## Setup Instructions

1. Install required packages:
```bash
pip install transformers torch sklearn pandas nltk sentence-transformers nlpaug prettytable
```

2. Download required NLTK data:
```python
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

3. Prepare your data:
   - Ensure your JIRA data is in CSV format with the following columns:
     - fields_summary
     - fields_description
     - fields_priority_name
     - fields_issuetype_name
     - fields_labels

## Usage

### Data Preprocessing
```python
from preprocessing import preprocess_text

# Load and preprocess your data
jira_data = pd.read_csv('your_data.csv')
jira_data['text_combined'] = jira_data['fields_summary'] + " " + jira_data['fields_description']
```

### Process a New Bug
```python
bug_description = "Your bug description here"
classification, priority, assigned_team = process_bug(bug_description)
print(f"Classification: {classification}")
print(f"Priority: {priority}")
print(f"Assigned Team: {assigned_team[0]}")
```

## Model Training

### Bug Classification Model
- Uses TF-IDF vectorization with 5000 features
- Implements Logistic Regression with increased iterations
- Includes comprehensive classification metrics

### Priority Prediction Model
- Fine-tunes BERT model
- Implements custom loss function with class weights
- Uses evaluation strategy with epoch-based checkpoints
- Supports model saving and loading

### Team Assignment Model
- Uses semantic similarity with predefined team descriptions
- Implements text augmentation for robust matching
- Provides similarity scores for all teams

## Performance Metrics

The system evaluates performance using:
- Precision, Recall, and F1-Score for classification
- Weighted metrics for priority prediction
- Cosine similarity scores for team assignment

## File Structure
```
.
├── results_priority/       # Priority model checkpoints
├── logs_priority/         # Training logs
├── bert_priority_model/   # Saved BERT model
├── tfidf_vectorizer.pkl   # Saved TF-IDF vectorizer
├── logistic_regression_model.pkl  # Saved classification model
└── preprocessed_jira_data.csv    # Preprocessed dataset
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Add your license information here]

## Authors
- Mohammad Bin Yousuf (CU# 101239019)
- Vrishab Prasanth Davey (UO# 300438343)
- Surendar Pala Dana Sekaran (UO#300401916)

## Acknowledgments
This project was developed as part of the CSI5137 - Applications of NLP and ML in Software Engineering course.

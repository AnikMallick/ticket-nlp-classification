# End-to-End Retrieval-Aware NLP Classification with Model Training and Error Analysis

## Project Overview

This project implements an end-to-end NLP classification system for support tickets, designed to mirror **real-world production ML workflows**.

The goal is not just to build a high-performing classifier, but to:

* Compare **classical ML**, **neural models**, and **retrieval-augmented approaches**
* Explicitly analyze **where and why models fail**
* Measure **when retrieval helps vs hurts**
* Bridge **ML modeling ↔ engineering deployment**

This repository is structured and executed in phases, each with clear deliverables and analysis, following industry best practices.

---

## Problem Statement

Given a short text document (support ticket), predict its **topic / intent category**.

This reflects common enterprise use cases such as:

* Customer support routing
* SLA risk detection
* Automated ticket triage

The focus is on **multi-class classification under class imbalance**, ambiguity, and overlapping semantics.

---

## Dataset

* **Input column**: `Document` (ticket text)
* **Target column**: `Topic_group` (intent / topic label)
* **Type**: Multi-class text classification (8 classes)

A publicly available support ticket dataset was used to simulate an internal enterprise ticket classification setting.
**Data Source:** [IT Support Ticket Topic Classifier](https://www.opendatabay.com/data/dataset/5e817530-63a1-43be-a7a7-8be1473afdbf)

---

## Repository Structure

```
ticket-nlp-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_tfidf_baselines.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── tfidf_logreg.py
│   │   └── tfidf_svm.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│
├── artifacts/
│   ├── tfidf_vectorizer_v01.pkl
│   ├── logreg_model_v01.pkl
│   └── svm_model_v01.pkl
│
├── api/        
│
├── README.md
└── requirements.txt
```

The codebase is structured so that **training, evaluation, retrieval, and deployment reuse the same core logic**.

---

## Phase 0 — Data Understanding

- Notebook: "notebooks/01_data_exploration.ipynb"

**Objective:** Understand data characteristics before modeling.

Performed analysis includes:

* Number of samples: 47837 and classes: 8.
* Class imbalance: The dataset is moderately imbalanced, with Hardware and HR Support dominating, motivating the use of macro-averaged evaluation metrics.

| Topic Group           | Sample Count |
| --------------------- | ------------ |
| Hardware              | 13,617       |
| HR Support            | 10,915       |
| Access                | 7,125        |
| Miscellaneous         | 7,060        |
| Storage               | 2,777        |
| Purchase              | 2,464        |
| Internal Project      | 2,119        |
| Administrative rights | 1,760        |

* Text length distribution: Most tickets are short to medium length, but a long tail of very large tickets exists, which can introduce noise and ambiguity for bag-of-words models.

| Statistic       | Value             |
| --------------- | ----------------- |
| Count           | 47,837            |
| Mean            | 291.88 characters |
| Std Dev         | 388.17            |
| Min             | 7                 |
| 25th Percentile | 110               |
| Median (50%)    | 175               |
| 75th Percentile | 304               |
| Max             | 7,015             |


This phase informs:

* Choice of macro-averaged metrics
* Later error and bias analysis

---

## Phase 1 — Classical ML Baselines (Completed)

- Notebook: "notebooks/02_tfidf_baselines.ipynb"

### Objective

Establish strong, interpretable baselines and understand their failure modes.

### Text Processing

* Lowercasing
* Removal of special characters
* No aggressive normalization (lemmatization/stopword removal deferred)

### Feature Engineering

* **TF-IDF** with uni-grams and bi-grams
* Sparse, high-dimensional representation
* Vectorizer: artifacts/tfidf_vectorizer_v01.pkl

### Models Trained

1. **Logistic Regression**

   * Loss: Negative Log Likelihood (cross-entropy)
   * Probabilistic outputs
   * Trained model: artifacts/logreg_model_v01.pkl

2. **Linear SVM**

   * Margin-based classifier
   * Strong baseline for sparse text features
   * Trained model: artifacts/svm_model_v01.pkl

### Evaluation Metrics

* Macro Precision
* Macro Recall
* Macro F1-score
* Confusion matrix
* Models were evaluated on a held-out test set (20%) using stratified splitting to
preserve class distribution.

Macro metrics are emphasized due to **class imbalance**.

### Results

| Model                        | Macro Precision | Macro Recall | Macro F1 |
| ---------------------------- | --------------- | ------------ | -------- |
| TF-IDF + Logistic Regression | 0.8993          | 0.8141       | 0.8478   |
| TF-IDF + Linear SVM          | 0.8772          | 0.8441       | 0.8594   |

#### Per-Class Recall Comparison

| Class                 | Logistic Regression | Linear SVM |
| --------------------- | ------------------- | ---------- |
| Access                | 85.8%               | **87.4%**  |
| Administrative rights | 61.3%               | **70.5%**  |
| HR Support            | **86.9%**           | 86.4%      |
| Hardware              | **91.5%**           | 88.5%      |
| Internal Project      | 77.1%               | **83.7%**  |
| Miscellaneous         | 82.4%               | **83.3%**  |
| Purchase              | 84.8%               | **87.4%**  |
| Storage               | 81.4%               | **87.9%**  |

Key observations:
- Administrative rights tickets are frequently misclassified as Hardware
- Hardware acts as a dominant class due to overlapping vocabulary and sample counts
- Linear SVM reduces confusion compared to Logistic Regression

Given our setting of ticket classification
- Missed tickets (low recall) are far more damaging
- Extra tickets in a queue (low precision) are usually manageable

For this reason, row-normalized confusion matrices were used as the primary
diagnostic tool for model comparison, enabling class-wise recall analysis
and identification of systematic misclassification patterns. 
Linear SVM performed better among the two models:
- Higher recall across most classes
- Reduced confusion with dominant categories
- Better robustness for semantically overlapping ticket types

### Key Observations

* TF-IDF baselines perform well for frequent, well-defined categories
* Significant confusion exists between semantically overlapping topics
* Minority classes exhibit lower recall
* Linear SVM generally improves separation for sparse features compared to Logistic Regression

### Artifacts Saved

* Trained models and vectorizer are serialized for reuse in later phases

---

## Phase 2 — Neural Model Training (WIP)

**Objective 1:** Evaluate whether a learned neural text encoder improves over TF-IDF baselines and analyze the effect of training duration vs early stopping on class-wise recall and confusion patterns.

- Notebook: "notebooks/03_neural_training.ipynb"

### Model Architecture

A lightweight neural classifier was trained from scratch:
* Tokenizer: Simple word-level tokenizer
* Vocabulary size: 30,000, With the given data extracted a vocabulary of length: 11608.
* Embedding dimension: 256 (trainable)
* Encoder: Mean pooling over token embeddings (mask-aware)
* Classifier: One hidden-layer MLP (256 units) with ReLU activation and Dropout (0.3)
* Loss: Cross-Entropy Loss
* Optimizer: Adam

### Training Setup

| Parameter        | Value                  |
| ---------------- | ---------------------- |
| Learning rate    | 0.0001                 |
| Batch size       | 32                     |
| Max epochs       | 50                     |
| Early stopping   | Optional               |
| Validation split | 20%                    |
| Stratified split | Yes                    |
| Label encoding   | `sklearn.LabelEncoder` |
| Padding length   | 256 tokens             |

**Two training regimes were compared:**

* Fixed training (50 epochs, no early stopping)
* Early stopping (triggered at epoch 31 based on validation Macro-F1)

### Evaluation Methodology

* Primary metric: Macro-averaged F1
* Secondary analysis: Row-normalized confusion matrix (recall per class)
* Rationale:
   - Dataset is imbalanced
   - Missed tickets (low recall) are more costly than extra tickets

Row-normalized confusion matrices were used to analyze where recall improves or degrades across training regimes.

### Results — Recall Analysis on hold out test data (Row-Normalized Confusion Matrix)

| Class                 | Epoch(50)           | Early Stopping: Epoch(31) |
| --------------------- | ------------------- | ---------- |
| Access                | 87.4%               | **87.8%**  |
| Administrative rights | 72.7%               | **73.0%**  |
| HR Support            | 85.3%               | **85.7%**  |
| Hardware              | **87.2%**           | 86.8%      |
| Internal Project      | **82.8%**           | 80.2%      |
| Miscellaneous         | 81.1%               | **81.9%**  |
| Purchase              | 87.6%               | **88.4%**  |
| Storage               | **86.6%**           | 86.1%      |
|                       |                     |            |
| F1 macro avg          | **0.8507**          | 0.8473     |

**Observations:**

* Administrative rights continues to show significant confusion with Hardware
* Hardware remains a dominant attractor class due to overlapping vocabulary
* Early stopping reduces overfitting getting overall better score, while overfitting does put more training on the majority classes.

Consistent confusion patterns observed across both settings:

* Administrative rights ↔ Hardware
* Internal Project ↔ Hardware/HR Support
* Miscellaneous ↔ Hardware
  
### Key Takeaways

* Neural embeddings do not drastically outperform TF-IDF on recall for this dataset
* Learned representations help smooth some class boundaries but cannot fully resolve semantic overlap
* Early stopping improves training efficiency but may slightly hurt overall recall
* These results motivate Phase 3 (Retrieval-Augmented Classification) to inject contextual grounding

### Artifacts Saved

* Early-stop model: artifacts/neural_model_bt_v01.pt
* Simple Word tokenizer vocab: artifacts/basic_tokenizer_v01.json
* Labelencoder: artifacts/labelencoder_neural_v01.pkl

---

## Phase 3 — Retrieval-Augmented Classification (Planned)

**Objective:** Measure the impact of retrieval on classification accuracy.

Planned steps:

* Build vector index (TF-IDF or learned embeddings)
* Retrieve top-k similar historical tickets
* Inject retrieved context into prediction pipeline

Evaluation includes:

* Performance with vs without retrieval
* Error types retrieval fixes
* Error types retrieval introduces

An **ablation study** is included to explicitly measure retrieval impact.

---

## Phase 4 — Error Analysis

This phase focuses on understanding model behavior rather than improving metrics.

Includes:

* Class-wise failure analysis
* Confusion trends
* Bias and imbalance effects
* Analysis of ambiguous / short tickets

Findings are documented in detail to demonstrate **production-grade ML thinking**.

---

## Phase 5 — Engineering & Deployment (Planned)

**Objective:** Bridge ML and systems.

Deployment plan:

* FastAPI inference service
* Async prediction endpoint
* Model and vectorizer loading from artifacts
* Simple caching for repeated queries

This phase demonstrates how trained ML models are **operationalized in real systems**.

---

## Key Takeaways

* Strong baselines are critical before adding complexity
* Neural and retrieval methods must be justified through analysis
* Error analysis is as important as accuracy
* Clean engineering enables reproducibility and deployment

---

## Tech Stack

* Python
* scikit-learn
* PyTorch
* FastAPI (Phase 5)
* NumPy, pandas

---

## Status

* Phase 1: ✅ Completed
* Phase 2: ⏳ In progress
* Phase 3–5: ⏳ Planned

---

This project is intentionally designed to reflect **real-world ML systems**, emphasizing rigor, analysis, and engineering discipline over model hype.

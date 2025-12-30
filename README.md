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

**Objective:** Understand data characteristics before modeling.

Notebook:
- `notebooks/01_data_exploration.ipynb`

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

### Objective

Establish strong, interpretable baselines and understand their failure modes.

Notebook:
- `notebooks/02_tfidf_baselines.ipynb`

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

## Phase 2 — Neural Model Training (Completed)

### Objective

Evaluate whether a **learned neural text encoder** improves over TF-IDF baselines and analyze the impact of **training duration vs early stopping** on class-wise recall and confusion patterns.

Notebook:
- `notebooks/03_neural_training.ipynb`
- `notebooks/04_neural_training_2.ipynb`

## Model Architecture

A lightweight neural text classifier was trained **from scratch** to serve as a minimal but interpretable neural baseline.

### Tokenization Experiments

Multiple tokenization strategies were evaluated:

- Word unigrams  
- Word bigrams  
- Word unigrams + bigrams  
- Character n-grams (3–5)  
- **Word unigrams + character n-grams (3–5)**

Among these, **Word Unigram + Character n-grams (3–5)** consistently achieved the best overall performance and recall balance.

### Architecture Details

- Vocabulary cap: 30,000  
  - Actual vocabulary size: **11,608** (Word unigrams) and **30,002** (Word unigrams + character n-grams (3–5))
- Embedding dimension: **256** (trainable)
- Encoder: Mean pooling over token embeddings (mask-aware)
- Classifier:
  - One hidden-layer MLP (256 units)
  - ReLU activation
  - Dropout: 0.3
- Loss: Cross-Entropy Loss
- Optimizer: Adam

## Training Setup

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

Two training regimes were compared:

- **Fixed training**: 50 epochs (no early stopping).
- **Early stopping**: Early stopping on validation Macro-F1 with delta `1e-4` and `patience=3`

## Evaluation Methodology

- **Primary metric**: Macro-averaged F1  
- **Secondary analysis**: Row-normalized confusion matrix (per-class recall)

**Rationale**

- Dataset is class-imbalanced
- Missed tickets (low recall) are more costly than extra tickets routed to a queue
- Row-normalized confusion matrices allow direct inspection of recall failures

## Results — Recall Analysis (Hold-out Test Set)

**Row-normalized confusion matrix (recall per class)**

| Class                 | Word Unigram<br>Epoch (50) | Word Unigram<br>Early Stopping (32) | Word Unigram + Char (3–5) |
| --------------------- | -------------------------- | ---------------------------------- | ------------------------- |
| Access                | 87.4%                      | 87.8%                               | **88.8%**                 |
| Administrative rights | 72.7%                      | 73.0%                               | **74.7%**                 |
| HR Support            | 85.3%                      | **85.7%**                           | 84.2%                     |
| Hardware              | **87.2%**                  | 86.8%                               | 87.0%                     |
| Internal Project      | **82.8%**                  | 80.2%                               | 81.4%                     |
| Miscellaneous         | 81.1%                      | 81.9%                               | **82.2%**                 |
| Purchase              | 87.6%                      | **88.4%**                           | 88.0%                     |
| Storage               | 86.6%                      | 86.1%                               | **88.5%**                 |
|                       |                            |                                     |                           |
| **Macro F1**          | 0.8507                     | 0.8473                              | **0.8539**                |

## Observations

- **Administrative rights** remains highly confused with **Hardware**, consistent with TF-IDF baselines
- **Hardware** acts as a dominant attractor class due to overlapping vocabulary and class frequency
- Character n-gram augmentation improves recall for several minority classes
- Early stopping reduces overfitting and improves training efficiency, with minor recall trade-offs

**Consistent confusion patterns across neural settings**

- Administrative rights ↔ Hardware  
- Internal Project ↔ Hardware / HR Support  
- Miscellaneous ↔ Hardware  

## Key Takeaways

- Neural embeddings do **not drastically outperform TF-IDF** for recall on this dataset
- Learned representations slightly smooth class boundaries but cannot fully resolve semantic overlap
- Early stopping acts primarily as a regularizer
- Character n-grams provide modest but consistent gains for ambiguous and minority classes
- These results motivate **Phase 3 — Retrieval-Augmented Classification**

### Artifacts Saved

* Early-stop model (Word Unigram): **artifacts/neural_model_btuni_v01.pt**
* Early-stop model (Word Unigram + Char (3–5)): **artifacts/neural_model_unicar3-5_v01.pt**
* Word Unigram tokenizer vocab: **artifacts/basic_tokenizer_uni_v01.json**
* Word Unigram + Char (3–5) tokenizer vocab: **artifacts/custom_tokenizer_unicar3-5_v01.json**
* Labelencoder: **artifacts/labelencoder_neural_v01.pkl**

## Phase 2 vs Phase 1 — Neural Model vs TF-IDF Baseline

Compared to the Phase 1 TF-IDF baseline, the neural encoder does **not deliver a step-change improvement in macro recall or F1**, despite increased modeling capacity and training cost. TF-IDF remains highly competitive on this dataset due to its strong lexical alignment with ticket language and class labels. The neural model shows **slightly smoother decision boundaries**, particularly when augmented with character n-grams, improving recall for some minority and noisy classes (e.g., *Administrative rights*, *Storage*). However, persistent confusion between semantically overlapping classes (e.g., *Hardware* vs *Administrative rights*) remains largely unchanged, indicating that **representation learning alone is insufficient** to resolve ambiguity inherent in short, underspecified ticket text.

---

## Phase 3 — Retrieval-Augmented Classification (RAC)

### Objective

Evaluate whether **injecting semantically similar historical tickets as additional context**
improves neural classification performance.

This phase explicitly tests a realistic production hypothesis:

> “If we retrieve relevant past tickets and append them as context, the classifier should perform better.”

The goal is not just performance gains, but to **measure when retrieval helps vs hurts**.

Notebook:
- `notebooks/05_retrieval_augmented_classification.ipynb`
- `notebooks/06_retrieval_augmented_classification_2.ipynb`
- `notebooks/07_retrieval_augmented_classification_3.ipynb`

### Retrieval Pipeline

A semantic retrieval pipeline was built using **pretrained sentence embeddings**.

#### Embedding Model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding type: Dense sentence embeddings
- Used only for retrieval (not fine-tuned)

#### Vector Index
- Library: FAISS
- Index types evaluated:
  - Cosine similarity
  - Euclidean distance
- Observation: Both indices produced nearly identical nearest neighbors

Based on this, **cosine similarity** was used for all experiments.

### Data Augmentation Strategy

For each ticket (train and test):

1. Generate sentence embedding
2. Retrieve top-`k` similar tickets from **training set**
3. Append retrieved documents as context: <original ticket text> [CONTEXT] <retrieved ticket 1> ... <retrieved ticket k>

### Retrieval-Augmented Data Flow

The following diagram illustrates the retrieval-augmented classification pipeline used in this phase:

┌──────────────┐
│ Ticket Text  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│ Sentence Transformer     │
│ (MiniLM-L6-v2 Embedding) │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ FAISS Vector Index       │
│ (Cosine / Euclidean)     │
└──────┬───────────────────┘
       │ Top-k Documents
       ▼
┌──────────────────────────┐
│ Context Augmentation     │
│ Original + Retrieved     │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Tokenizer                │
│ (Word / Char n-grams)    │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Neural Classifier        │
│ (Embedding → MLP)        │
└──────────┬───────────────┘
           │
           ▼
     Predicted Topic

#### Important Design Choices

- `k = 5` for test data
- For training data:
  - `k = 6` retrieved
  - Top result (self-match) removed
  - Final effective context size = 5
- **Only document text was retrieved**
  - Labels were NOT injected
  - Prevents label leakage
- Both **training and test data** were augmented

This setup mirrors real-world RAC systems without oracle access.

### Classification Model (Reused from Phase 2)

The same neural architecture from Phase 2 was reused to isolate the effect of retrieval.

**Architecture:**
- Trainable embedding layer
- Mean pooling encoder
- Classifier:
  - Linear (256) → ReLU → Dropout (0.3) → Linear (num classes)
- Loss: Cross-Entropy
- Optimizer: Adam

**Tokenization Experiments:**
1. Word Unigram
2. Word Unigram + Character n-grams (3–5)

### Engineering Constraints & Optimization Challenges

The Word + Char (3–5) tokenizer significantly increased sequence length.

Hardware constraint:
- GPU: GTX 1660 Super (6GB VRAM)

Mitigation strategy:
- Model kept on GPU
- Input batches dynamically moved:
  - CPU → GPU (forward + backward)
  - GPU → CPU (after step)
- Enabled training completion but increased wall-clock time

This reflects real-world tradeoffs between model complexity and infrastructure.

### Results — Retrieval-Augmented Neural Models

#### Observed Outcome

Despite correct retrieval, **retrieval-augmented training degraded performance**:

- Validation Macro-F1 plateaued around **0.50–0.52**
- Training slowed significantly
- Confusion increased across nearly all classes

#### Row-Normalized Confusion Matrix (Recall)

Performance collapsed relative to Phase 2, with strong confusion toward dominant classes (Hardware, Miscellaneous).

Key failure patterns:
- Administrative rights ↔ Hardware
- Internal Project ↔ Hardware / HR Support
- Miscellaneous ↔ Hardware

This degradation was observed for:
- Word Unigram
- Word Unigram + Char (3–5)

### Retrieval Quality Verification (Sanity Check)

To ensure retrieval itself was not the issue, a **retrieval-only baseline** was evaluated:

#### Method
- Retrieve top-10 similar training tickets
- Predict label via majority vote
- Tested with:
  - Cosine similarity index
  - Euclidean distance index

#### Observation
- Both indices produced nearly identical confusion matrices
- Retrieval-only voting achieved **reasonable recall** across most classes
- Confirms:
  - Vector DB is correct
  - Nearest neighbors are semantically meaningful

This isolates the failure to **retrieval + neural encoder interaction**, not retrieval quality.

### Retrieval-Only vs RAC vs Phase 2 — Comparative Analysis

A clear contrast emerges when comparing **retrieval-only voting**, **retrieval-augmented classification (RAC)**, and the **Phase 2 neural classifier**.

Retrieval-only voting using semantic neighbors performs reasonably well because it preserves label locality and avoids representation collapse. Phase 2’s neural classifier, trained solely on the original ticket text, achieves the strongest overall Macro-F1 by learning stable class boundaries without external noise. In contrast, retrieval-augmented classification degrades performance because retrieved context is injected without any mechanism to model relevance, leading to semantic dilution and optimization instability. This demonstrates that retrieval is most effective either as a **decision-level signal (voting / reranking)** or when paired with architectures explicitly designed to reason over retrieved evidence, rather than naively concatenated into the input.

### Analysis — Why Retrieval Hurt Performance

The failure is architectural, not algorithmic.

1. **Naïve context concatenation**
   - Retrieved tickets often contain mixed or conflicting intents
   - No distinction between query and context tokens

2. **Mean pooling has no selectivity**
   - All tokens contribute equally
   - Retrieved context dominates representation

3. **Sequence length explosion**
   - Signal from original ticket diluted
   - Optimization becomes unstable

4. **Model capacity mismatch**
   - Shallow MLP cannot reason over multi-document context
   - No attention, gating, or relevance weighting

### Key Takeaways

- Retrieval correctness ≠ retrieval usefulness
- Naïve RAC can significantly degrade performance
- Neural encoders must be **retrieval-aware**, not retrieval-blind
- TF-IDF + SVM remains a surprisingly strong baseline
- This mirrors real-world RAG failures in production systems

### Artifacts

The following artifacts were persisted for reproducibility and analysis:

- **Sentence Transformer (Embedding Model):**  
  Not saved separately. The model (`sentence-transformers/all-MiniLM-L6-v2`) was used strictly as a frozen feature extractor with no fine-tuning. Reproducibility is ensured by recording the model name and version.

- **FAISS Vector Indexes:**  
  Saved for semantic retrieval during experimentation and analysis. Both cosine similarity and Euclidean distance indexes were evaluated and produced comparable results.
  Artifacts: **artifacts/traindata_similarity_index_v01.index**, **artifacts/traindata_euclidian_index_v01.index**

- **Training Corpus & Labels:**  
  Persisted to maintain alignment between:
  - FAISS index positions
  - Training text samples
  - Ground-truth labels  
  Artifacts: **artifacts/rac_corpus_similarity-euclidian_index_v01.json**

  This ensures correct label mapping during retrieval, voting, and sanity checks.

- **Retrieval-augmented datasets:** 
  For ease of experimentation the augmented data was saved
  **data/processed/agumented_ticketdata_similarity_v01.json**, **data/processed/agumented_ticketdata_euclidian_v01.json**

- **Neural RAC Models (Experimental):**  
  Not saved. Retrieval-augmented neural classifiers showed degraded performance compared to Phase 2 baselines, and therefore were not retained as deployable artifacts.

- **Evaluation Outputs:**  
  Confusion matrices and classification reports were saved for all retrieval strategies to support comparative analysis in our notebooks.

### Transition to Next Phase

Phase 3 demonstrates that **retrieval must be integrated thoughtfully**.

Potential next directions:
- Late fusion (separate encoding of query and context)
- Attention over retrieved documents
- Retrieval-based re-ranking instead of concatenation

These insights motivate Phase 4: **Error Analysis & Failure Attribution**.

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

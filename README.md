# SMS Spam Detection using NLP

An end-to-end Natural Language Processing (NLP) project for detecting whether an SMS message is **spam** or **ham (legitimate)**.

This project compares a traditional machine-learning baseline using **TF-IDF + Logistic Regression** with an advanced **DistilBERT** transformer model. The final DistilBERT model is integrated into an interactive **Streamlit** application.

---

## Project Overview

Spam messages are unwanted or suspicious messages that may contain advertisements, scams, fake rewards, malicious links, or requests for personal information.

The goal of this project is to build an NLP classification system that automatically classifies an SMS message as:

- **HAM** — legitimate/non-spam message
- **SPAM** — unwanted or suspicious message

The project follows an end-to-end machine-learning workflow:

1. Data preparation and cleaning
2. Text preprocessing
3. Baseline NLP model
4. Advanced transformer model
5. Model evaluation and comparison
6. Model deployment with Streamlit
7. Responsible AI analysis

---

## Dataset

The project uses the **SMS Spam Collection** dataset.

The original dataset contains SMS messages labelled as either `ham` or `spam`.

### Dataset Columns

| Column | Description |
|---|---|
| `label` | Message category: ham or spam |
| `message` | SMS message text |

The original dataset contained **5,572 records**.

After removing duplicate messages, the dataset contained **5,169 records**.

### Class Distribution

- Ham: **4,516**
- Spam: **653**

The dataset is therefore imbalanced, with legitimate messages occurring much more frequently than spam messages.

Because of this imbalance, accuracy alone was not used to evaluate the models. Precision, recall, F1-score, and the confusion matrix were also considered.

---

## Data Preparation

The data preparation process included:

- Selecting the relevant message and label columns
- Removing duplicate messages
- Encoding labels:
  - `ham = 0`
  - `spam = 1`
- Checking for missing values
- Cleaning unnecessary HTML-like tags and whitespace
- Converting text to lowercase
- Preserving useful spam-related information such as numbers, punctuation, URLs, and currency symbols

The data was divided using a **70/15/15 stratified split**:

- 70% training
- 15% validation
- 15% testing

Stratification was used to preserve the ham/spam class distribution across the splits.

---

## Baseline Model

The baseline model uses traditional NLP techniques.

### TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** was used to convert SMS messages into numerical feature vectors.

The implementation used:

- Unigrams and bigrams
- `min_df=2`
- Sublinear TF scaling
- Maximum of 10,000 features

The resulting TF-IDF representation contained **8,863 features**.

### Logistic Regression

A Logistic Regression classifier was trained using the TF-IDF features.

### Baseline Test Results

| Metric | Score |
|---|---:|
| Accuracy | 96.13% |
| Precision | 100.00% |
| Recall | 69.39% |
| F1-score | 81.93% |

The baseline achieved very high precision but lower recall.

On the test set:

- False positives: **0**
- False negatives: **30**

---

## Advanced Model — DistilBERT

The advanced NLP model uses **DistilBERT**, a transformer-based language model.

The pretrained `distilbert-base-uncased` model was fine-tuned for binary SMS classification.

### Tokenization

Messages were tokenized using the DistilBERT tokenizer.

Configuration included:

- Maximum sequence length: 128 tokens
- Truncation enabled
- Padding enabled

### Fine-Tuning

The model was fine-tuned using:

- 3 training epochs
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Training batch size: 16
- Evaluation batch size: 32
- Best model selected using validation F1-score

Training was performed using a Tesla T4 GPU in Google Colab.

The best validation performance occurred at **epoch 2**.

---

## DistilBERT Test Results

| Metric | Score |
|---|---:|
| Accuracy | 99.10% |
| Precision | 96.91% |
| Recall | 95.92% |
| F1-score | 96.41% |

Test-set confusion matrix:

```text
                 Predicted
                 HAM   SPAM

Actual HAM       675     3
Actual SPAM        4    94
# Cost-Sensitive Real-Time Fraud Detection System

A production-oriented fraud detection system designed for **extreme class imbalance**, **low-latency inference**, and **cost-sensitive decision making** in financial transactions.

---

## 🚀 Overview

Fraud detection systems in financial domains face three core challenges:

* Extreme class imbalance (~0.13% fraud cases)
* Strict latency constraints (<50 ms inference)
* Asymmetric cost of errors (false negatives are far more expensive than false positives)

This project focuses on building a **practical machine learning system** that balances predictive performance with real-world business cost.

---

## 📂 Dataset

This project uses the PaySim synthetic dataset:

👉 https://www.kaggle.com/datasets/mtalaltariq/paysim-data

### Dataset Description

* Simulates mobile money transactions over a 30-day period
* Based on real financial transaction patterns
* Includes injected fraudulent behavior for evaluation

### Setup

Download the dataset and place it in:

```
data/paysim.csv
```

---

## ⚙️ Installation

```bash
git clone https://github.com/krishnakg12/fraud-detection-ml-system.git
cd fraud-detection-ml-system

pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Ensure dataset is placed at:

```
data/paysim.csv
```

2. Update dataset path in code (if required):

```python
DATA_PATH = "data/paysim.csv"
```

3. Run the model:

```bash
python src/model.py
```

---

## 🧠 Methodology

### Cost-Sensitive Learning

* Optimizes decision threshold based on financial loss
* Reduces impact of high-cost false negatives

### Graph-Based Feature Engineering

* Degree centrality (hub detection)
* Transaction pair frequency
* Historical risk scoring (time-aware, no leakage)

### Model

* XGBoost (GPU/CPU supported)
* Handles imbalance using `scale_pos_weight`

### Drift Detection

* Page-Hinkley algorithm
* Detects distribution shifts in streaming scenarios

---

## 📊 Results

| Metric         | Value   |
| -------------- | ------- |
| Recall (Fraud) | 93.7%   |
| Precision      | 66.8%   |
| F1 Score       | 78.0%   |
| AUC-ROC        | 0.984   |
| Latency        | < 50 ms |

---

## 🏗️ System Architecture

```
Data Layer → Feature Engineering → ML Model → API Serving
```

* Data Layer → ingestion and preprocessing
* ML Layer → model training and threshold optimization
* Serving Layer → real-time inference (FastAPI-ready)

---

## 📁 Repository Structure

```
fraud-detection-ml-system/
│
├── src/
│   └── model.py
├── data/              # dataset (not included)
├── results/           # outputs (plots, metrics)
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 🛠️ Tech Stack

* Python
* XGBoost
* Scikit-learn
* Pandas / NumPy
* Matplotlib
* FastAPI

---

## 🔍 Key Insights

* Accuracy alone is misleading for fraud detection
* Cost-sensitive optimization improves real-world outcomes
* Graph features provide efficient relational signals
* Latency constraints strongly influence model design

---

## 🚧 Future Work

* Transformer-based sequence modeling
* Graph Neural Networks (GNNs) under latency constraints
* Online learning for real-time adaptation

---

## 👤 Author

Krishna K G
B.E. Artificial Intelligence & Machine Learning (2026)

# Cost-Sensitive Real-Time Fraud Detection System

This repository contains research and implementation of a real-time fraud detection system designed under extreme class imbalance and strict latency constraints.

## Overview

Fraud detection in financial systems presents three major challenges:

* Extreme class imbalance (~0.13% fraud rate)
* Real-time inference constraints (<50 ms latency)
* Asymmetric cost of errors (false negatives far more expensive than false positives)

This work focuses on designing a system that balances detection performance with operational and economic constraints.

---

## Research Contributions

### 1. Cost-Sensitive Learning Framework

* Formulated fraud detection as a cost optimization problem instead of pure classification
* Optimized decision thresholds based on financial loss rather than accuracy

### 2. Hybrid Graph-Based Feature Engineering

* Avoided computationally expensive Graph Neural Networks
* Extracted relational signals (degree, PageRank) from transaction graphs
* Integrated graph features into XGBoost for efficient inference

### 3. Real-Time ML System Design

* Built a microservice-based inference pipeline using FastAPI
* Achieved sub-50 ms latency for real-time predictions
* Designed for scalability and streaming transaction environments

---

## Results

| Metric         | Value   |
| -------------- | ------- |
| Recall (Fraud) | 93.7%   |
| Precision      | 66.8%   |
| F1 Score       | 78.0%   |
| AUC-ROC        | 0.984   |
| Latency        | < 50 ms |

* Detected 1,154 fraudulent transactions
* Prevented ~$1.15M in estimated financial losses

---

## System Architecture

The system follows a three-tier architecture:

1. Data Layer:

   * Transaction ingestion
   * Feature enrichment (graph + temporal)

2. ML Layer:

   * XGBoost model with cost-sensitive training
   * Threshold optimization module

3. Serving Layer:

   * FastAPI-based inference service
   * Real-time prediction and alert generation

---

## Repository Structure

* `src/` → Core ML pipeline (feature engineering, training, inference)
* `experiments/` → Research experiments and analysis
* `papers/` → Research papers (CIMA 2025, ICNARI 2026)
* `results/` → Metrics and evaluation outputs

---

## Papers

1. **AI-Powered Payment Fraud Detection and Optimization System**

   * Accepted at CIMA 2025 (NIT Puducherry)
   * Publication scheduled: April 2026

2. **A Systematic Review of AI-Powered Payment Fraud Detection**

   * Accepted at ICNARI 2026 (NIT Patna)
   * Analysis of 120+ research papers across ML, deep learning, and graph-based systems

---

## Key Insights

* Pure accuracy-based optimization is insufficient for real-world fraud systems
* Graph-based features can approximate relational learning without GNN overhead
* Cost-sensitive thresholding significantly improves business outcomes
* Latency constraints fundamentally shape model design choices

---

## Tech Stack

* Python
* XGBoost
* FastAPI
* NetworkX (graph features)
* Docker (deployment-ready)

---

## Future Work

* Integration with transformer-based sequence models
* Exploration of Graph Neural Networks under optimized latency constraints
* Adaptive learning for concept drift handling

---

## Author

Krishna K G
B.E. Artificial Intelligence & Machine Learning (2026)

---

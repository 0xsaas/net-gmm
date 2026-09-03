# 📥 Data Ingestion & Dataset Conditioning

> **Upstream**: [[Architecture Overview]]  
> **Downstream**: [[Feature Engineering & PCA]]  
> **Source Module**: `data/download_data.py`

This module manages the automated retrieval, verification, and data partitioning of the canonical **NSL-KDD Network Intrusion Dataset**.

---

## 📊 Dataset Specifications

The NSL-KDD benchmark eliminates redundant records from KDD Cup 1999 to prevent classifier bias.

- **Training Records (`KDDTrain+.txt`)**: 125,973 network flow records.
- **Testing Records (`KDDTest+.txt`)**: 22,544 network flow records.
- **Feature Dimensionality**: 41 network telemetry attributes + 1 ground truth label + 1 difficulty score.

---

## ⚖️ Unsupervised Baseline Constraint

To simulate real-world cyber defense where zero-day attacks are unknown a priori:
1. The training baseline is strictly filtered to contain **only records with `label == 'normal'`** (67,343 samples, 53.5% of train set).
2. The GMM density estimator is trained exclusively on normal behavior to learn the baseline statistical manifold.
3. In validation and testing, attacks are treated as out-of-distribution events.

---

## 🛡️ Fallback Generator
If external mirrors are unreachable, `generate_synthetic_nsl_kdd()` programmatically creates an identical 43-column synthetic dataset exhibiting log-normal, Poisson, and categorical distributions for continuous integration environments.

---

## 🔗 Connected Notes
- [[Architecture Overview]]
- [[Feature Engineering & PCA]]
- [[GMM Density Engine]]

# 📊 Model Evaluation & Benchmark Metrics

> **Upstream**: [[Threshold Optimization]]  
> **Source Module**: `main.py` / `src/utils.py`  
> **Artifacts**: `outputs/evaluation_report.json` | `outputs/confusion_matrix.png` | `outputs/pca_clusters_2d.png`

Comprehensive benchmark evaluation of the unsupervised GMM + PCA anomaly detection pipeline on **15,781 holdout test telemetry flows** from the NSL-KDD dataset.

---

## 🏆 Final Benchmark Telemetry

| Performance Metric | Holdout Test Score | Description |
| :--- | :---: | :--- |
| **Optimal Gaussian Components ($K^*$)** | **9** | Selected via Bayesian Information Criterion |
| **PCA Latent Components** | **48** | $\ge 95.04\%$ cumulative variance retained |
| **Decision Boundary ($\epsilon^*$)** | **90.1418** | $F_1$-calibrated log-likelihood cutoff |
| **Anomaly Precision** | **92.34%** | Accuracy of raised security alerts |
| **Anomaly Recall** | **93.31%** | Proportion of network attacks detected |
| **Anomaly $F_1$-Score** | **92.82%** | Harmonic mean of precision and recall |
| **False Positive Rate (FPR)** | **10.22%** | False alarms on benign traffic |
| **ROC-AUC Score** | **0.9652** | Area under the ROC curve |
| **PR-AUC Score** | **0.9582** | Area under the Precision-Recall curve |

---

## 🗃️ Confusion Matrix

```
                      Predicted Normal (0)    Predicted Attack (1)
True Normal (0)              6,103 (89.8%)            695 (10.2%)
True Attack (1)                601 ( 6.7%)          8,382 (93.3%)
```

- **True Negatives (TN)**: 6,103 correctly cleared benign flows.
- **False Positives (FP)**: 695 benign flows flagged for investigation.
- **False Negatives (FN)**: 601 missed attacks.
- **True Positives (TP)**: **8,382** network attacks intercepted.

---

## 🖼️ Visualizations

### 2D Principal Component Space & GMM Centroids
![[pca_clusters_2d.png]]

### Confusion Matrix Heatmap
![[confusion_matrix.png]]

---

## 🔗 Connected Notes
- [[Threshold Optimization]]
- [[GMM Density Engine]]
- [[Real-Time Streaming Engine]]
- [[Architecture Overview]]

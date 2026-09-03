# 🎯 Threshold Optimization & Decision Theory

> **Upstream**: [[GMM Density Engine]]  
> **Downstream**: [[Model Evaluation & Metrics]] | [[Real-Time Streaming Engine]]  
> **Source Module**: `src/threshold_tuner.py`  
> **Visual Output**: `outputs/score_distribution.png`

This module calibrates the log-likelihood decision threshold $\epsilon^*$ based on statistical decision theory to maximize the anomaly detection $F_1$-score.

---

## ⚖️ Decision Rule

Given the log-likelihood score $\ln p(x)$ from the [[GMM Density Engine]]:
$$\hat{y}(x) = \begin{cases} 1 \ (\text{Attack / Anomaly}), & \text{if } \ln p(x) < \epsilon \\ 0 \ (\text{Normal Baseline}), & \text{if } \ln p(x) \ge \epsilon \end{cases}$$

---

## 📈 $F_1$-Score Maximization

The threshold $\epsilon$ is scanned across percentile ranges ($[0.1\%, 15.0\%]$) on validation telemetry:
$$\epsilon^* = \arg\max_\epsilon F_1(\epsilon) = \arg\max_\epsilon \frac{2 \cdot \text{Precision}(\epsilon) \cdot \text{Recall}(\epsilon)}{\text{Precision}(\epsilon) + \text{Recall}(\epsilon)}$$

### Calibrated Boundary
- **Optimal Threshold ($\epsilon^*$)**: **`90.1418`**
- **Validation Precision**: 91.79%
- **Validation Recall**: 92.68%
- **Validation $F_1$-Score**: 92.23%
- **Validation False Positive Rate (FPR)**: 10.95%

---

## 📊 Score Distribution Plot
![[score_distribution.png]]

---

## 🔗 Connected Notes
- [[GMM Density Engine]]
- [[Model Evaluation & Metrics]]
- [[Real-Time Streaming Engine]]
- [[Architecture Overview]]

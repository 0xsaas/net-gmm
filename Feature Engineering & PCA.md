# 🔬 Feature Engineering & Principal Component Analysis (PCA)

> **Upstream**: [[Data Ingestion]]  
> **Downstream**: [[GMM Density Engine]]  
> **Source Module**: `src/preprocess.py`  
> **Artifact**: `models/preprocessor.joblib`

This module transforms raw, heterogeneous network flow telemetry into a clean, orthogonal latent feature space suitable for Gaussian density estimation.

---

## 📐 Mathematical Transformations

### 1. Skewness Mitigation ($\ln(1 + x)$)
Continuous attributes like `duration`, `src_bytes`, `dst_bytes`, `count`, and `srv_count` exhibit extreme right-skewness spanning multiple orders of magnitude. We apply the natural logarithm plus one:
$$\tilde{x} = \ln(1 + x)$$
This compresses long-tailed distributions and prevents covariance collapse during GMM fitting.

### 2. Categorical One-Hot Encoding
Nominal features (`protocol_type`, `service`, `flag`) are encoded using `OneHotEncoder(handle_unknown='ignore')`, expanding categorical values into binary indicator columns.

### 3. Standardization (Z-Score)
All numerical and encoded columns are centered and scaled based strictly on training normal baseline statistics:
$$z = \frac{\tilde{x} - \mu_{\text{normal}}}{\sigma_{\text{normal}}}$$

### 4. Dimensionality Reduction (PCA $\ge 95\%$ Cumulative Variance)
Given the standardized covariance matrix $C = \frac{1}{N} Z^T Z = V \Lambda V^T$:
$$\frac{\sum_{i=1}^d \lambda_i}{\sum_{j=1}^D \lambda_j} \ge 0.95$$

In our training run:
- **Original Dimensions**: 122 features (continuous + one-hot categories).
- **PCA Retained Dimensions ($d$)**: **48 orthogonal components**.
- **Retained Cumulative Variance**: **95.04%**.

---

## ⚡ Online Single-Sample Transformation
The class exposes `transform_single(dict)` for micro-batch online inference in the [[Real-Time Streaming Engine]].

---

## 🔗 Connected Notes
- [[Data Ingestion]]
- [[GMM Density Engine]]
- [[Architecture Overview]]

# 🧠 Gaussian Mixture Model (GMM) Density Engine

> **Upstream**: [[Feature Engineering & PCA]]  
> **Downstream**: [[Threshold Optimization]]  
> **Source Module**: `src/gmm_engine.py`  
> **Artifact**: `models/gmm_model.joblib`  
> **Visual Output**: `outputs/bic_aic_elbow.png`

The **GMM Density Engine** models the multi-modal probability density function of normal network traffic in the PCA latent space using multivariate Gaussian distributions.

---

## 📐 Mathematical Formulation

The probability density of a sample $x \in \mathbb{R}^d$ is:
$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

Where:
- $K$: Number of Gaussian mixture components (clusters).
- $\pi_k$: Prior mixture weights ($\sum \pi_k = 1, \pi_k \ge 0$).
- $\mu_k \in \mathbb{R}^d$: Cluster mean centroid.
- $\Sigma_k \in \mathbb{R}^{d \times d}$: Full covariance matrix capturing intra-cluster correlations.

---

## 🔍 Model Selection: BIC & AIC

To discover the optimal cluster count $K \in [1, 10]$ without manual guessing, the engine computes:
$$\text{BIC} = -2 \ln \hat{L} + p \ln(N)$$
$$\text{AIC} = -2 \ln \hat{L} + 2p$$

Where $p = K - 1 + Kd + K \frac{d(d+1)}{2}$ is the number of free parameters for full covariance matrices.

### Cluster Search Results

| $K$ | BIC Score | AIC Score | Convergence |
| :---: | :---: | :---: | :---: |
| 1 | +9,802,230 | +9,791,070 | True |
| 2 | -302,380 | -324,708 | True |
| 3 | -3,110,624 | -3,144,122 | True |
| 4 | -15,984,289 | -16,028,956 | True |
| 5 | -13,426,718 | -13,482,554 | True |
| 6 | -16,603,822 | -16,670,827 | True |
| 7 | -18,951,655 | -19,029,828 | True |
| 8 | -19,498,099 | -19,587,442 | True |
| **9** | **-20,864,183** | **-20,964,695** | **Optimal ($K^*$)** |
| 10 | -20,601,957 | -20,713,638 | True |

**Selected Topology**: $K^* = 9$ components.

---

## 📈 BIC/AIC Elbow Visualization
![[bic_aic_elbow.png]]

---

## 🔗 Connected Notes
- [[Feature Engineering & PCA]]
- [[Threshold Optimization]]
- [[Model Evaluation & Metrics]]
- [[Architecture Overview]]

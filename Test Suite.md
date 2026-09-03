# 🧪 Automated Test Suite & Validation

> **Upstream**: [[Architecture Overview]]  
> **Source Module**: `tests/test_pipeline.py`

The test suite validates data ingestion, preprocessor pipelines, PCA variance preservation, GMM component search, threshold tuning, and streaming alert generation using `pytest`.

---

## 📋 Test Matrix

| Test Function | Target Component | Validation Objective |
| :--- | :--- | :--- |
| `test_data_schema` | [[Data Ingestion]] | Verifies canonical 43-column schema and synthetic generation integrity. |
| `test_preprocessor_pipeline` | [[Feature Engineering & PCA]] | Verifies log1p transforms, scaling, PCA variance retention ($\ge 95\%$), single-sample inference, and joblib serialization. |
| `test_gmm_engine` | [[GMM Density Engine]] | Validates BIC/AIC grid search ($K=1..3$), full covariance EM fitting, and sample log-likelihood scoring. |
| `test_threshold_tuner` | [[Threshold Optimization]] | Tests percentile threshold scanning and $F_1$-score maximization logic. |
| `test_stream_simulator` | [[Real-Time Streaming Engine]] | Tests live packet processing, threshold alerting, and ISO-8601 JSON alert output schema. |

---

## ⚡ Execution Command

```bash
conda activate net_gmm_env
PYTHONPATH=. pytest -v tests/test_pipeline.py
```

```
============================= test session starts ==============================
tests/test_pipeline.py::test_data_schema PASSED                          [ 20%]
tests/test_pipeline.py::test_preprocessor_pipeline PASSED                [ 40%]
tests/test_pipeline.py::test_gmm_engine PASSED                           [ 60%]
tests/test_pipeline.py::test_threshold_tuner PASSED                      [ 80%]
tests/test_pipeline.py::test_stream_simulator PASSED                     [100%]

============================== 5 passed in 1.93s ===============================
```

---

## 🔗 Connected Notes
- [[Architecture Overview]]
- [[Data Ingestion]]
- [[Feature Engineering & PCA]]
- [[GMM Density Engine]]
- [[Threshold Optimization]]
- [[Real-Time Streaming Engine]]

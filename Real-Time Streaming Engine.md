# ⚡ Real-Time Streaming Engine & Threat Alerting

> **Upstream**: [[Threshold Optimization]] | [[Feature Engineering & PCA]]  
> **Source Module**: `src/stream_simulator.py`  
> **Alerts Log**: `outputs/alerts.json`

The **Real-Time Streaming Engine** simulates live network packet telemetry ingestion, executes micro-batch inference against the pre-trained [[Feature Engineering & PCA]] and [[GMM Density Engine]] artifacts, and emits structured security alerts.

---

## ⚙️ Ingestion & Detection Flow

1. **Packet Arrival**: Telemetry arrives as a key-value dictionary (e.g., protocol, service, byte counts, rates).
2. **Online Projection**: Transformed through `preprocessor.transform_single(dict)` into the 48-dimensional PCA space.
3. **Log-Likelihood Scoring**: Evaluated via `gmm.score_single(vec)` to obtain density score $\ln p(x)$.
4. **Threshold Comparison**: If $\ln p(x) < \epsilon^*$, a security alert is emitted.

---

## 🚨 Security Alert Schema (`outputs/alerts.json`)

```json
{
  "timestamp": "2026-09-03T10:43:13.081457+00:00",
  "status": "ANOMALY_DETECTED",
  "log_likelihood": -427.5485,
  "threshold": 90.1418,
  "raw_features": {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "private",
    "flag": "S0",
    "src_bytes": 0,
    "dst_bytes": 0,
    "count": 244,
    "srv_count": 2,
    "serror_rate": 1.0,
    "label": "neptune"
  },
  "ground_truth_label": "neptune"
}
```

---

## 📊 Live Streaming Performance
- **Throughput**: ~25.5 packets / second (configurable inter-packet delay).
- **Latency**: < 2 ms per packet transformation and density evaluation.

---

## 🔗 Connected Notes
- [[Threshold Optimization]]
- [[Model Evaluation & Metrics]]
- [[Architecture Overview]]

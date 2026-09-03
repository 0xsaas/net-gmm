"""
Real-Time MLOps Network Streaming Simulator and Security Alert Emitter.
Simulates packet-by-packet traffic ingestion, real-time feature transformation,
GMM density evaluation, and ISO-8601 JSON alert emission.
"""

import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.preprocess import NSLKDDPreprocessor
from src.gmm_engine import GMMAnomalyDetector

logger = logging.getLogger("StreamSimulator")


class StreamSimulator:
    """
    Simulates real-time network packet telemetry ingestion.
    Performs micro-batch inference and emits structured JSON security alerts.
    """

    def __init__(
        self,
        preprocessor: NSLKDDPreprocessor,
        gmm_model: GMMAnomalyDetector,
        threshold: float,
        alert_log_path: str = "outputs/alerts.json",
        packet_delay: float = 0.02,
    ):
        self.preprocessor = preprocessor
        self.gmm_model = gmm_model
        self.threshold = threshold
        self.alert_log_path = alert_log_path
        self.packet_delay = packet_delay
        
        self.total_packets_processed: int = 0
        self.anomalies_detected: int = 0
        self.emitted_alerts: List[Dict[str, Any]] = []
        
        os.makedirs(os.path.dirname(self.alert_log_path), exist_ok=True)
        # Clear or initialize alert file
        with open(self.alert_log_path, "w") as f:
            f.write("")

    def _emit_alert(
        self,
        log_likelihood: float,
        raw_features: Dict[str, Any],
        actual_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format and append structured ISO-8601 security alert to disk and console."""
        # ISO-8601 UTC timestamp
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        
        # Clean non-serializable types from raw features
        sanitized_raw = {}
        for k, v in raw_features.items():
            if isinstance(v, (np.integer, int)):
                sanitized_raw[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                sanitized_raw[k] = round(float(v), 4)
            else:
                sanitized_raw[k] = str(v)

        alert_payload = {
            "timestamp": timestamp_iso,
            "status": "ANOMALY_DETECTED",
            "log_likelihood": round(float(log_likelihood), 4),
            "threshold": round(float(self.threshold), 4),
            "raw_features": sanitized_raw,
        }
        
        if actual_label is not None:
            alert_payload["ground_truth_label"] = actual_label

        self.emitted_alerts.append(alert_payload)
        
        # Append line-delimited JSON
        with open(self.alert_log_path, "a") as f:
            f.write(json.dumps(alert_payload) + "\n")
            
        return alert_payload

    def process_packet(
        self,
        packet_dict: Dict[str, Any],
        verbose: bool = True,
    ) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        """
        Process a single incoming network packet dictionary in real time.
        
        Args:
            packet_dict: Raw telemetry dictionary.
            verbose: Whether to log alerts to console.
            
        Returns:
            Tuple of (is_anomaly, log_likelihood, alert_payload_or_None).
        """
        actual_label = packet_dict.get("label", None)
        
        # 1. Transform single sample to PCA space
        pca_vec = self.preprocessor.transform_single(packet_dict)
        
        # 2. Score under GMM
        log_lh = self.gmm_model.score_single(pca_vec)
        
        self.total_packets_processed += 1
        is_anomaly = log_lh < self.threshold
        alert = None

        if is_anomaly:
            self.anomalies_detected += 1
            alert = self._emit_alert(log_lh, packet_dict, actual_label)
            if verbose:
                protocol = packet_dict.get("protocol_type", "N/A")
                service = packet_dict.get("service", "N/A")
                src_bytes = packet_dict.get("src_bytes", 0)
                dst_bytes = packet_dict.get("dst_bytes", 0)
                logger.warning(
                    f"🚨 [SECURITY ALERT] Packet #{self.total_packets_processed:05d} "
                    f"| Log-Likelihood: {log_lh:.2f} < {self.threshold:.2f} "
                    f"| Protocol: {protocol} | Service: {service} "
                    f"| Bytes: {src_bytes}->{dst_bytes} | GroundTruth: {actual_label}"
                )
        return is_anomaly, log_lh, alert

    def run_stream(
        self,
        df_stream: pd.DataFrame,
        max_packets: Optional[int] = 50,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run continuous or bounded streaming simulation over DataFrame records.
        
        Args:
            df_stream: Test DataFrame containing incoming packet stream.
            max_packets: Maximum number of packets to simulate (None for all).
            verbose: Whether to print real-time alert logs.
            
        Returns:
            Summary dictionary with stream statistics.
        """
        n_records = len(df_stream) if max_packets is None else min(len(df_stream), max_packets)
        logger.info(
            f"Starting Real-Time Packet Stream Simulator: "
            f"{n_records} packets | Delay: {self.packet_delay:.3f}s/pkt | "
            f"Alert Destination: {self.alert_log_path}"
        )

        start_time = time.time()
        
        for idx in range(n_records):
            row_dict = df_stream.iloc[idx].to_dict()
            self.process_packet(row_dict, verbose=verbose)
            if self.packet_delay > 0:
                time.sleep(self.packet_delay)

        total_time = max(time.time() - start_time, 1e-6)
        throughput = self.total_packets_processed / total_time
        anomaly_rate = (self.anomalies_detected / max(self.total_packets_processed, 1)) * 100

        summary = {
            "total_packets_processed": self.total_packets_processed,
            "anomalies_detected": self.anomalies_detected,
            "anomaly_rate_percent": round(anomaly_rate, 2),
            "elapsed_seconds": round(total_time, 2),
            "throughput_pkts_per_sec": round(throughput, 2),
            "alerts_file": self.alert_log_path,
        }

        logger.info(
            f"Stream simulation completed: Processed {self.total_packets_processed} packets in {total_time:.2f}s "
            f"({throughput:.1f} pkts/sec) | Anomalies Detected: {self.anomalies_detected} ({anomaly_rate:.1f}%)"
        )
        return summary

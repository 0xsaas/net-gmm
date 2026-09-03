"""
Net-GMM Anomaly Detection Pipeline Package.
Provides modules for feature engineering, GMM modeling, threshold tuning, and real-time streaming.
"""

from src.preprocess import NSLKDDPreprocessor
from src.gmm_engine import GMMAnomalyDetector
from src.threshold_tuner import ThresholdTuner
from src.stream_simulator import StreamSimulator
from src.utils import (
    setup_logger,
    plot_bic_aic_elbow,
    plot_score_distribution,
    plot_pca_clusters_2d,
    plot_confusion_matrix,
    save_json_report,
)

__all__ = [
    "NSLKDDPreprocessor",
    "GMMAnomalyDetector",
    "ThresholdTuner",
    "StreamSimulator",
    "setup_logger",
    "plot_bic_aic_elbow",
    "plot_score_distribution",
    "plot_pca_clusters_2d",
    "plot_confusion_matrix",
    "save_json_report",
]

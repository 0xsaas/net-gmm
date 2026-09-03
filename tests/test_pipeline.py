"""
Unit and Integration Test Suite for NSL-KDD GMM & PCA Anomaly Detection Project.
Tests data generation, preprocessing, PCA variance retention, GMM fitting, threshold tuning,
and real-time streaming alert generation.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from data.download_data import generate_synthetic_nsl_kdd, load_raw_dataset, COLUMN_NAMES
from src.preprocess import NSLKDDPreprocessor
from src.gmm_engine import GMMAnomalyDetector
from src.threshold_tuner import ThresholdTuner
from src.stream_simulator import StreamSimulator


@pytest.fixture
def synthetic_data(tmp_path):
    """Fixture to generate temporary synthetic train and test datasets."""
    train_file = str(tmp_path / "KDDTrain+.txt")
    test_file = str(tmp_path / "KDDTest+.txt")
    
    generate_synthetic_nsl_kdd(train_file, n_samples=300, is_train=True)
    generate_synthetic_nsl_kdd(test_file, n_samples=100, is_train=False)
    
    df_train = load_raw_dataset(train_file)
    df_test = load_raw_dataset(test_file)
    return df_train, df_test, tmp_path


def test_data_schema(synthetic_data):
    """Verify that dataset schema contains canonical 43 NSL-KDD columns."""
    df_train, df_test, _ = synthetic_data
    assert len(df_train.columns) == 43
    assert list(df_train.columns) == COLUMN_NAMES
    assert "label" in df_train.columns
    assert len(df_train) == 300
    assert len(df_test) == 100


def test_preprocessor_pipeline(synthetic_data):
    """Test log-transforms, one-hot encoding, scaling, and PCA variance retention."""
    df_train, df_test, tmp_path = synthetic_data
    
    # Filter normal baseline for training
    normal_mask = df_train["label"].astype(str).str.startswith("normal")
    df_train_normal = df_train[normal_mask]
    
    preprocessor = NSLKDDPreprocessor(variance_threshold=0.95, random_state=42)
    X_train_pca = preprocessor.fit_transform(df_train_normal)
    
    assert preprocessor.is_fitted
    assert X_train_pca.shape[0] == len(df_train_normal)
    assert X_train_pca.shape[1] >= 2
    
    # Cumulative explained variance must meet or exceed threshold
    cum_var = float(np.sum(preprocessor.pca.explained_variance_ratio_))
    assert cum_var >= 0.95 or np.isclose(cum_var, 0.95, atol=1e-2)
    
    # Transform test set
    X_test_pca = preprocessor.transform(df_test)
    assert X_test_pca.shape[0] == len(df_test)
    assert X_test_pca.shape[1] == X_train_pca.shape[1]
    
    # Test single sample transform
    single_sample = df_test.iloc[0].to_dict()
    vec_1d = preprocessor.transform_single(single_sample)
    assert vec_1d.shape == (X_train_pca.shape[1],)
    
    # Test serialization
    model_path = str(tmp_path / "preprocessor.joblib")
    preprocessor.save(model_path)
    loaded_prep = NSLKDDPreprocessor.load(model_path)
    assert loaded_prep.is_fitted
    np.testing.assert_allclose(vec_1d, loaded_prep.transform_single(single_sample))


def test_gmm_engine(synthetic_data):
    """Test GMM component selection with BIC/AIC and sample scoring."""
    df_train, df_test, tmp_path = synthetic_data
    df_train_normal = df_train[df_train["label"].astype(str).str.startswith("normal")]
    
    prep = NSLKDDPreprocessor(variance_threshold=0.95, random_state=42).fit(df_train_normal)
    X_train = prep.transform(df_train_normal)
    
    detector = GMMAnomalyDetector(min_k=1, max_k=3, random_state=42, n_init=1)
    optimal_k, bic_dict, aic_dict = detector.find_optimal_k(X_train)
    
    assert 1 <= optimal_k <= 3
    assert len(bic_dict) == 3
    assert len(aic_dict) == 3
    
    detector.fit(X_train, optimal_k=optimal_k)
    assert detector.is_fitted
    
    # Test score_samples
    scores = detector.score_samples(X_train)
    assert scores.shape == (len(X_train),)
    assert np.all(np.isfinite(scores))
    
    # Test score_single
    single_score = detector.score_single(X_train[0])
    assert isinstance(single_score, float)
    assert np.isclose(single_score, scores[0])


def test_threshold_tuner(synthetic_data):
    """Test log-likelihood threshold optimization for F1 maximization."""
    df_train, df_test, _ = synthetic_data
    df_train_normal = df_train[df_train["label"].astype(str).str.startswith("normal")]
    
    prep = NSLKDDPreprocessor(variance_threshold=0.95, random_state=42).fit(df_train_normal)
    X_train = prep.transform(df_train_normal)
    X_test = prep.transform(df_test)
    
    detector = GMMAnomalyDetector(min_k=2, max_k=2, random_state=42, n_init=1).fit(X_train, optimal_k=2)
    test_scores = detector.score_samples(X_test)
    y_test = prep.get_ground_truth_labels(df_test)
    
    tuner = ThresholdTuner(min_percentile=0.5, max_percentile=20.0, n_steps=20)
    thresh, best_metrics = tuner.tune(test_scores, y_test)
    
    assert isinstance(thresh, float)
    assert "f1_score" in best_metrics
    assert 0.0 <= best_metrics["f1_score"] <= 1.0
    
    eval_metrics = tuner.evaluate(test_scores, y_test, threshold=thresh)
    assert "confusion_matrix" in eval_metrics
    assert eval_metrics["tp"] + eval_metrics["fp"] + eval_metrics["tn"] + eval_metrics["fn"] == len(df_test)


def test_stream_simulator(synthetic_data):
    """Test packet-by-packet streaming simulation and ISO-8601 JSON alert output."""
    df_train, df_test, tmp_path = synthetic_data
    df_train_normal = df_train[df_train["label"].astype(str).str.startswith("normal")]
    
    prep = NSLKDDPreprocessor(variance_threshold=0.95, random_state=42).fit(df_train_normal)
    X_train = prep.transform(df_train_normal)
    
    detector = GMMAnomalyDetector(min_k=2, max_k=2, random_state=42, n_init=1).fit(X_train, optimal_k=2)
    # Set high threshold to guarantee some alerts fire
    forced_thresh = float(np.percentile(detector.score_samples(X_train), 50.0))
    
    alerts_file = str(tmp_path / "alerts.json")
    simulator = StreamSimulator(
        preprocessor=prep,
        gmm_model=detector,
        threshold=forced_thresh,
        alert_log_path=alerts_file,
        packet_delay=0.0,
    )
    
    summary = simulator.run_stream(df_test, max_packets=10, verbose=False)
    assert summary["total_packets_processed"] == 10
    assert os.path.exists(alerts_file)
    
    if summary["anomalies_detected"] > 0:
        with open(alerts_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == summary["anomalies_detected"]
        first_alert = json.loads(lines[0])
        assert first_alert["status"] == "ANOMALY_DETECTED"
        assert "timestamp" in first_alert
        assert "log_likelihood" in first_alert
        assert "threshold" in first_alert
        assert "raw_features" in first_alert

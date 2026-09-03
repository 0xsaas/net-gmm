"""
End-to-End Orchestration Pipeline for NSL-KDD Network Anomaly Detection.
Integrates Data Ingestion -> Skewness Mitigation & PCA -> GMM Density Estimation ->
Validation Threshold Optimization -> Test Evaluation -> Visualizations -> Real-Time Streaming.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.download_data import download_nsl_kdd, load_raw_dataset
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

logger = setup_logger("MainPipeline")


def parse_args():
    """Parse command-line arguments for flexible pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Production Unsupervised Network Anomaly Detection using GMM and PCA on NSL-KDD."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset files.")
    parser.add_argument("--download", action="store_true", help="Force programmatic download of NSL-KDD.")
    parser.add_argument("--variance-threshold", type=float, default=0.95, help="PCA cumulative variance ratio (>= 0.95).")
    parser.add_argument("--min-k", type=int, default=1, help="Minimum Gaussian components to evaluate.")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum Gaussian components to evaluate.")
    parser.add_argument("--val-ratio", type=float, default=0.30, help="Fraction of test set used for threshold tuning.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for plots and reports.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for saving serialized models.")
    parser.add_argument("--stream-packets", type=int, default=30, help="Number of packets for real-time streaming simulation.")
    parser.add_argument("--stream-delay", type=float, default=0.03, help="Inter-packet delay (seconds) in streaming simulator.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def run_pipeline(args):
    """Execute the complete end-to-end anomaly detection pipeline."""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print("=" * 80)
    print(" 🚀 UNSUPERVISED NETWORK ANOMALY DETECTION: GMM + PCA ON NSL-KDD")
    print("=" * 80)

    # ---------------------------------------------------------
    # STEP 1: DATA INGESTION & DATASET SPLITTING
    # ---------------------------------------------------------
    logger.info(">>> STEP 1: Ingesting NSL-KDD Dataset...")
    train_path, test_path = download_nsl_kdd(data_dir=args.data_dir, force_download=args.download)
    
    df_train_raw = load_raw_dataset(train_path)
    df_test_raw = load_raw_dataset(test_path)
    
    logger.info(f"Raw Train Set: {df_train_raw.shape[0]:,} samples | Raw Test Set: {df_test_raw.shape[0]:,} samples")

    # Unsupervised Constraint: Training set must strictly contain ONLY normal baseline traffic
    train_normal_mask = df_train_raw["label"].astype(str).str.strip().str.lower().str.startswith("normal")
    df_train_normal = df_train_raw[train_normal_mask].copy().reset_index(drop=True)
    
    logger.info(
        f"Filtered Training Baseline (Normal only): {len(df_train_normal):,} samples "
        f"({(len(df_train_normal)/len(df_train_raw))*100:.1f}% of train set)"
    )

    # Split test set into Validation (for threshold tuning) and Holdout Test (for final benchmark)
    df_val, df_test_holdout = train_test_split(
        df_test_raw,
        test_size=(1.0 - args.val_ratio),
        random_state=args.random_state,
        stratify=df_test_raw["label"].astype(str).str.strip().str.lower().str.startswith("normal"),
    )
    logger.info(
        f"Partitioned Test Telemetry: Validation set = {len(df_val):,} samples | "
        f"Holdout Test set = {len(df_test_holdout):,} samples"
    )

    # ---------------------------------------------------------
    # STEP 2: LOG-TRANSFORMATION, SCALING & PCA PIPELINE
    # ---------------------------------------------------------
    logger.info(">>> STEP 2: Executing Log-Transforms, StandardScaler & PCA Pipeline...")
    preprocessor = NSLKDDPreprocessor(
        variance_threshold=args.variance_threshold,
        random_state=args.random_state,
    )
    
    # Fit strictly on normal baseline
    X_train_pca = preprocessor.fit_transform(df_train_normal)
    X_val_pca = preprocessor.transform(df_val)
    X_test_pca = preprocessor.transform(df_test_holdout)

    # Save preprocessor artifact
    prep_save_path = os.path.join(args.model_dir, "preprocessor.joblib")
    preprocessor.save(prep_save_path)

    # Ground truth labels (0 = Normal, 1 = Attack)
    y_val = preprocessor.get_ground_truth_labels(df_val)
    y_test = preprocessor.get_ground_truth_labels(df_test_holdout)

    logger.info(
        f"Transformed Shapes -> Train: {X_train_pca.shape} | Val: {X_val_pca.shape} | Test: {X_test_pca.shape}"
    )

    # ---------------------------------------------------------
    # STEP 3: STATISTICAL GMM ENGINE & BIC/AIC SELECTION
    # ---------------------------------------------------------
    logger.info(f">>> STEP 3: Grid Searching GMM Component Topology K in [{args.min_k}, {args.max_k}]...")
    gmm_detector = GMMAnomalyDetector(
        min_k=args.min_k,
        max_k=args.max_k,
        covariance_type="full",
        random_state=args.random_state,
    )
    
    # Evaluate information criteria across K
    optimal_k, bic_dict, aic_dict = gmm_detector.find_optimal_k(X_train_pca)
    
    # Fit final production model
    gmm_detector.fit(X_train_pca, optimal_k=optimal_k)
    
    # Save model artifact
    gmm_save_path = os.path.join(args.model_dir, "gmm_model.joblib")
    gmm_detector.save(gmm_save_path)

    # Plot BIC/AIC Elbow
    bic_plot_path = os.path.join(args.output_dir, "bic_aic_elbow.png")
    plot_bic_aic_elbow(bic_dict, aic_dict, optimal_k=optimal_k, save_path=bic_plot_path)
    logger.info(f"[Plot Saved] BIC/AIC Elbow curve: {bic_plot_path}")

    # ---------------------------------------------------------
    # STEP 4: DECISION THEORY & THRESHOLD TUNING (F1 MAXIMIZATION)
    # ---------------------------------------------------------
    logger.info(">>> STEP 4: Tuning Decision Boundary Epsilon on Validation Telemetry...")
    val_scores = gmm_detector.score_samples(X_val_pca)
    train_normal_scores = gmm_detector.score_samples(X_train_pca)

    tuner = ThresholdTuner(min_percentile=0.1, max_percentile=15.0, n_steps=250)
    optimal_threshold, val_metrics = tuner.tune(
        val_scores=val_scores,
        y_val=y_val,
        baseline_normal_scores=train_normal_scores,
    )

    # ---------------------------------------------------------
    # STEP 5: FINAL HOLDOUT TEST SET EVALUATION
    # ---------------------------------------------------------
    logger.info(">>> STEP 5: Evaluating Model on Unseen Holdout Test Telemetry...")
    test_scores = gmm_detector.score_samples(X_test_pca)
    test_metrics = tuner.evaluate(scores=test_scores, y_true=y_test, threshold=optimal_threshold)
    y_test_pred = (test_scores < optimal_threshold).astype(int)

    # Print summary dashboard
    print("\n" + "=" * 80)
    print(" 📊 NSL-KDD NETWORK ANOMALY DETECTION: FINAL EVALUATION REPORT")
    print("=" * 80)
    print(f" • Optimal Gaussian Components (K*):    {optimal_k}")
    print(f" • PCA Components (≥ 95% Variance):     {X_train_pca.shape[1]}")
    print(f" • Optimal Decision Threshold (ε*):     {optimal_threshold:.4f}")
    print(f" • Test Anomaly Precision:              {test_metrics['precision'] * 100:.2f}%")
    print(f" • Test Anomaly Recall:                 {test_metrics['recall'] * 100:.2f}%")
    print(f" • Test F1-Score:                       {test_metrics['f1_score'] * 100:.2f}%")
    print(f" • Test False Positive Rate (FPR):      {test_metrics['fpr'] * 100:.2f}%")
    print(f" • Test ROC-AUC Score:                  {test_metrics['roc_auc']:.4f}")
    print(f" • Test PR-AUC Score:                   {test_metrics['pr_auc']:.4f}")
    print("\n[Confusion Matrix Breakdown]:")
    print(f"   True Normal (TN):   {test_metrics['tn']:,}   |  False Anomaly (FP): {test_metrics['fp']:,}")
    print(f"   Missed Attack (FN): {test_metrics['fn']:,}   |  Detected Attack (TP): {test_metrics['tp']:,}")
    print("\n[Detailed Classification Report]:\n" + test_metrics["classification_report"])
    print("=" * 80)

    # ---------------------------------------------------------
    # STEP 6: PUBLICATION-GRADE VISUALIZATIONS & METRICS EXPORT
    # ---------------------------------------------------------
    logger.info(">>> STEP 6: Generating Evaluation Visualizations and Metric Reports...")
    
    # 1. Score Distribution
    test_norm_scores = test_scores[y_test == 0]
    test_att_scores = test_scores[y_test == 1]
    score_dist_path = os.path.join(args.output_dir, "score_distribution.png")
    plot_score_distribution(
        normal_scores=test_norm_scores,
        attack_scores=test_att_scores,
        threshold=optimal_threshold,
        save_path=score_dist_path,
    )
    logger.info(f"[Plot Saved] Log-likelihood Score Distribution: {score_dist_path}")

    # 2. 2D PCA Cluster Space
    pca_plot_path = os.path.join(args.output_dir, "pca_clusters_2d.png")
    plot_pca_clusters_2d(
        X_pca=X_test_pca,
        y_true=y_test,
        y_pred=y_test_pred,
        centroids=gmm_detector.means_,
        save_path=pca_plot_path,
    )
    logger.info(f"[Plot Saved] 2D PCA Cluster Scatter: {pca_plot_path}")

    # 3. Confusion Matrix
    cm_plot_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true=y_test, y_pred=y_test_pred, save_path=cm_plot_path)
    logger.info(f"[Plot Saved] Confusion Matrix Heatmap: {cm_plot_path}")

    # 4. Save JSON Report
    full_report = {
        "pipeline_config": vars(args),
        "optimal_k": optimal_k,
        "optimal_threshold_epsilon": optimal_threshold,
        "bic_scores": bic_dict,
        "aic_scores": aic_dict,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    report_json_path = os.path.join(args.output_dir, "evaluation_report.json")
    save_json_report(full_report, report_json_path)
    logger.info(f"[Report Saved] JSON Evaluation Summary: {report_json_path}")

    # ---------------------------------------------------------
    # STEP 7: REAL-TIME MLOPS STREAMING SIMULATOR
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(" ⚡ REAL-TIME MLOPS PACKET INGESTION & SECURITY ALERT STREAMING")
    print("=" * 80)
    
    alerts_log_path = os.path.join(args.output_dir, "alerts.json")
    simulator = StreamSimulator(
        preprocessor=preprocessor,
        gmm_model=gmm_detector,
        threshold=optimal_threshold,
        alert_log_path=alerts_log_path,
        packet_delay=args.stream_delay,
    )

    stream_summary = simulator.run_stream(
        df_stream=df_test_holdout,
        max_packets=args.stream_packets,
        verbose=True,
    )

    print("\n[Streaming Simulation Summary]:")
    for k, v in stream_summary.items():
        print(f" • {k}: {v}")
    print("=" * 80 + "\n")
    logger.info("Pipeline execution successfully completed!")


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(cli_args)

"""
Utility Helpers for Network Anomaly Detection System.
Provides publication-grade visualizations (BIC/AIC elbow, score distributions, 2D PCA clusters),
structured logging configuration, and JSON metric reporting.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend for headless and production runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configure visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


def setup_logger(name: str = "NetGMM", log_level: int = logging.INFO) -> logging.Logger:
    """Configure a unified, formatted console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


def plot_bic_aic_elbow(
    bic_dict: Dict[int, float],
    aic_dict: Dict[int, float],
    optimal_k: int,
    save_path: str = "outputs/bic_aic_elbow.png",
) -> str:
    """
    Generate BIC and AIC curve across candidate Gaussian components K.
    Marks the mathematically optimal cluster count K*.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    k_vals = sorted(list(bic_dict.keys()))
    bics = [bic_dict[k] for k in k_vals]
    aics = [aic_dict[k] for k in k_vals]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    ax.plot(k_vals, bics, marker="o", linewidth=2.2, color="#1f77b4", label="BIC (Bayesian Information Criterion)")
    ax.plot(k_vals, aics, marker="s", linewidth=2.2, linestyle="--", color="#ff7f0e", label="AIC (Akaike Information Criterion)")
    
    # Highlight optimal K
    opt_bic = bic_dict[optimal_k]
    ax.scatter([optimal_k], [opt_bic], s=180, color="#d62728", zorder=5, label=f"Optimal Topology (K* = {optimal_k})")
    ax.axvline(x=optimal_k, color="#d62728", linestyle=":", alpha=0.7)

    ax.set_title("GMM Cluster Topology Selection: BIC & AIC vs. Component Count (K)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Gaussian Components (K)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Information Criterion Score (Lower is Better)", fontsize=11, fontweight="semibold")
    ax.set_xticks(k_vals)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_score_distribution(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    threshold: float,
    save_path: str = "outputs/score_distribution.png",
) -> str:
    """
    Generate density histogram and KDE comparing Normal vs. Attack log-likelihood distributions.
    Draws a vertical decision threshold line at epsilon.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Subsample if extremely large for smooth plotting
    max_pts = 20000
    if len(normal_scores) > max_pts:
        norm_sub = np.random.choice(normal_scores, size=max_pts, replace=False)
    else:
        norm_sub = normal_scores

    if len(attack_scores) > max_pts:
        att_sub = np.random.choice(attack_scores, size=max_pts, replace=False)
    else:
        att_sub = attack_scores

    # Determine sensible plot range (filter extreme -inf outliers for visualization clarity)
    p1 = np.percentile(np.concatenate([norm_sub, att_sub]), 0.5)
    p99 = np.percentile(np.concatenate([norm_sub, att_sub]), 99.5)
    
    norm_filt = norm_sub[(norm_sub >= p1) & (norm_sub <= p99)]
    att_filt = att_sub[(att_sub >= p1) & (att_sub <= p99)]

    sns.histplot(
        norm_filt, bins=60, kde=True, stat="density",
        color="#2ca02c", label="Normal Baseline (Class 0)", alpha=0.45, ax=ax
    )
    sns.histplot(
        att_filt, bins=60, kde=True, stat="density",
        color="#d62728", label="Network Attack / Anomaly (Class 1)", alpha=0.45, ax=ax
    )

    ax.axvline(
        x=threshold, color="#111111", linestyle="--", linewidth=2.2,
        label=f"Decision Boundary (ε = {threshold:.2f})"
    )

    ax.set_title("Log-Likelihood Score Distribution: Normal vs. Anomaly", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Log-Likelihood ln p(x)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_pca_clusters_2d(
    X_pca: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    save_path: str = "outputs/pca_clusters_2d.png",
    max_points: int = 4000,
) -> str:
    """
    Generate 2D PCA scatter projection showing normal points, anomalies, and GMM cluster centroids.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Subsample for rendering performance
    n_samples = len(X_pca)
    if n_samples > max_points:
        idx = np.random.choice(n_samples, size=max_points, replace=False)
        X_sub = X_pca[idx]
        y_true_sub = y_true[idx]
        y_pred_sub = y_pred[idx]
    else:
        X_sub = X_pca
        y_true_sub = y_true
        y_pred_sub = y_pred

    # Scatter plot: Normal (True Negative / True Normal)
    norm_mask = (y_pred_sub == 0)
    anom_mask = (y_pred_sub == 1)

    ax.scatter(
        X_sub[norm_mask, 0], X_sub[norm_mask, 1],
        c="#1f77b4", alpha=0.4, s=20, edgecolors="none", label="Predicted Normal (ln p(x) ≥ ε)"
    )
    ax.scatter(
        X_sub[anom_mask, 0], X_sub[anom_mask, 1],
        c="#d62728", alpha=0.6, s=30, edgecolors="black", linewidths=0.5, label="Predicted Anomaly (ln p(x) < ε)"
    )

    # Plot Centroids (First 2 PCA dimensions)
    if centroids is not None:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c="#ffeb3b", marker="X", s=200, edgecolors="black", linewidths=1.5,
            zorder=10, label="GMM Component Centroids (μ_k)"
        )

    ax.set_title("2D Principal Component Space: Anomaly Separation & GMM Centroids", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Principal Component 1", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Principal Component 2", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "outputs/confusion_matrix.png",
) -> str:
    """Plot normalized and count confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues", cbar=True,
        xticklabels=["Pred Normal (0)", "Pred Attack (1)"],
        yticklabels=["True Normal (0)", "True Attack (1)"],
        ax=ax, annot_kws={"size": 11, "fontweight": "bold"}
    )

    ax.set_title("Network Anomaly Detection: Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Ground Truth Class", fontsize=10, fontweight="semibold")
    ax.set_xlabel("GMM Predicted Class", fontsize=10, fontweight="semibold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def save_json_report(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary report to JSON formatted with clean indentation."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Recursively convert numpy types to native Python types
    def _convert(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    clean_data = _convert(data)
    with open(filepath, "w") as f:
        json.dump(clean_data, f, indent=2)

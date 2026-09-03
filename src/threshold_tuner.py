"""
Decision Theory and Threshold Optimization Module.
Finds the optimal log-likelihood decision boundary (epsilon) to maximize F1-score
for binary network anomaly classification (0 = Normal, 1 = Attack).
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

logger = logging.getLogger("ThresholdTuner")


class ThresholdTuner:
    """
    Optimizes log-likelihood threshold epsilon based on decision-theoretic F1 maximization.
    Samples with score < epsilon are classified as Anomalies (1); otherwise Normal (0).
    """

    def __init__(
        self,
        min_percentile: float = 0.1,
        max_percentile: float = 10.0,
        n_steps: int = 200,
    ):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.n_steps = n_steps
        
        self.optimal_threshold: Optional[float] = None
        self.best_metrics: Dict[str, Any] = {}
        self.tuning_history: Optional[pd.DataFrame] = None

    def tune(
        self,
        val_scores: np.ndarray,
        y_val: np.ndarray,
        baseline_normal_scores: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Scan percentile-based and value-based candidate thresholds to find epsilon* maximizing F1-Score.
        
        Args:
            val_scores: Log-likelihood scores on validation set containing both normal & attack samples.
            y_val: Binary ground-truth labels (0 = Normal, 1 = Attack).
            baseline_normal_scores: Optional normal-only scores to derive percentile anchors.
            
        Returns:
            Tuple of (optimal_threshold, best_metrics_dictionary).
        """
        logger.info(
            f"Scanning thresholds across percentiles [{self.min_percentile}%, {self.max_percentile}%] "
            f"over {len(val_scores)} validation samples..."
        )

        reference_scores = baseline_normal_scores if baseline_normal_scores is not None else val_scores
        percentiles = np.linspace(self.min_percentile, self.max_percentile, self.n_steps)
        candidate_thresholds = np.percentile(reference_scores, percentiles)
        
        # Also include min-to-median linear threshold search for comprehensive coverage
        score_range = np.linspace(np.percentile(val_scores, 0.05), np.percentile(val_scores, 50.0), self.n_steps)
        all_candidates = np.unique(np.concatenate([candidate_thresholds, score_range]))
        all_candidates = np.sort(all_candidates)

        history_rows: List[Dict[str, Any]] = []
        best_f1 = -1.0
        best_thresh = float(all_candidates[0])
        best_dict: Dict[str, Any] = {}

        for thresh in all_candidates:
            # Anomaly condition: log_likelihood < threshold => class 1 (Attack)
            y_pred = (val_scores < thresh).astype(int)
            
            # Precision, Recall, F1
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            row = {
                "threshold": float(thresh),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "fpr": float(fpr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
            history_rows.append(row)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                best_dict = row

        self.tuning_history = pd.DataFrame(history_rows)
        self.optimal_threshold = best_thresh
        self.best_metrics = best_dict

        logger.info(
            f"Threshold optimization complete: Optimal threshold epsilon* = {self.optimal_threshold:.4f} | "
            f"Validation F1 = {best_dict['f1_score'] * 100:.2f}% | "
            f"Precision = {best_dict['precision'] * 100:.2f}% | "
            f"Recall = {best_dict['recall'] * 100:.2f}% | "
            f"FPR = {best_dict['fpr'] * 100:.2f}%"
        )
        return self.optimal_threshold, self.best_metrics

    def evaluate(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate binary classification performance on an arbitrary test set given threshold epsilon.
        
        Args:
            scores: Log-likelihood scores.
            y_true: Ground truth binary labels (0 = Normal, 1 = Attack).
            threshold: Log-likelihood decision threshold (defaults to self.optimal_threshold).
            
        Returns:
            Dictionary containing precision, recall, f1, auc_roc, fpr, and confusion matrix.
        """
        thresh = threshold if threshold is not None else self.optimal_threshold
        if thresh is None:
            raise RuntimeError("Threshold is not set. Run tune() first or provide an explicit threshold.")

        y_pred = (scores < thresh).astype(int)
        
        # Invert scores for ROC-AUC because lower score = higher anomaly likelihood
        anomaly_probability_proxy = -scores
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        try:
            roc_auc = float(roc_auc_score(y_true, anomaly_probability_proxy))
        except Exception:
            roc_auc = 0.0
            
        try:
            pr_auc = float(average_precision_score(y_true, anomaly_probability_proxy))
        except Exception:
            pr_auc = 0.0

        report_str = classification_report(y_true, y_pred, target_names=["Normal (0)", "Attack (1)"], digits=4)

        metrics = {
            "threshold": float(thresh),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "fpr": float(fpr),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_str,
        }
        return metrics

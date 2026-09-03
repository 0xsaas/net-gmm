"""
Gaussian Mixture Model (GMM) Statistical Anomaly Detection Engine.
Performs automated cluster topology selection via BIC/AIC grid search and fits
an Expectation-Maximization (EM) full covariance Gaussian density estimator.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

logger = logging.getLogger("GMMEngine")


class GMMAnomalyDetector:
    """
    Statistical density estimator using full-covariance Gaussian Mixture Models.
    Computes exact log-likelihoods ln p(x) under multi-modal normal distributions.
    """

    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 10,
        covariance_type: str = "full",
        random_state: int = 42,
        max_iter: int = 200,
        n_init: int = 3,
    ):
        self.min_k = min_k
        self.max_k = max_k
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        
        self.gmm: Optional[GaussianMixture] = None
        self.optimal_k: Optional[int] = None
        self.bic_scores: Dict[int, float] = {}
        self.aic_scores: Dict[int, float] = {}
        self.models_cache: Dict[int, GaussianMixture] = {}
        self.is_fitted: bool = False

    def find_optimal_k(self, X: np.ndarray) -> Tuple[int, Dict[int, float], Dict[int, float]]:
        """
        Evaluate candidate Gaussian components K in [min_k, max_k] using BIC and AIC.
        
        Mathematical Formulation:
            BIC = -2 ln L_hat + p * ln(N)
            AIC = -2 ln L_hat + 2 * p
            where p is the number of free parameters and N is sample size.
            
        Args:
            X: Standardized and PCA-reduced feature matrix.
            
        Returns:
            Tuple of (optimal_k, bic_dict, aic_dict).
        """
        logger.info(f"Initiating GMM component selection across K in [{self.min_k}, {self.max_k}]...")
        self.bic_scores.clear()
        self.aic_scores.clear()
        self.models_cache.clear()

        for k in range(self.min_k, self.max_k + 1):
            gmm_candidate = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            gmm_candidate.fit(X)
            
            bic = float(gmm_candidate.bic(X))
            aic = float(gmm_candidate.aic(X))
            
            self.bic_scores[k] = bic
            self.aic_scores[k] = aic
            self.models_cache[k] = gmm_candidate
            
            logger.info(
                f"[K={k:02d}] BIC: {bic:12.2f} | AIC: {aic:12.2f} | "
                f"Converged: {gmm_candidate.converged_} (iter: {gmm_candidate.n_iter_})"
            )

        # Select K minimizing BIC (strongly penalizes overparameterization)
        self.optimal_k = min(self.bic_scores, key=self.bic_scores.get)
        logger.info(
            f"Component search complete. Optimal cluster count selected: K* = {self.optimal_k} "
            f"(Min BIC = {self.bic_scores[self.optimal_k]:.2f})"
        )
        return self.optimal_k, self.bic_scores, self.aic_scores

    def fit(self, X: np.ndarray, optimal_k: Optional[int] = None) -> "GMMAnomalyDetector":
        """
        Fit final GMM density estimator with optimal K components.
        
        Args:
            X: Preprocessed normal baseline feature matrix.
            optimal_k: Optional manual override for component count.
        """
        if optimal_k is None:
            if self.optimal_k is None:
                self.find_optimal_k(X)
            optimal_k = self.optimal_k
        else:
            self.optimal_k = optimal_k

        logger.info(
            f"Fitting production GaussianMixture(n_components={optimal_k}, "
            f"covariance_type='{self.covariance_type}') using EM algorithm..."
        )
        
        # If model already fitted during search, retrieve or refit with higher n_init
        self.gmm = GaussianMixture(
            n_components=optimal_k,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=max(self.n_init, 5),
            random_state=self.random_state,
        )
        self.gmm.fit(X)
        self.is_fitted = True
        logger.info(
            f"GMM training converged in {self.gmm.n_iter_} iterations with lower-bound log-likelihood: "
            f"{self.gmm.lower_bound_:.4f}"
        )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate per-sample log-likelihood ln p(x) under the fitted Gaussian Mixture.
        
        Mathematical Formulation:
            ln p(x) = ln sum_{k=1}^K pi_k * N(x | mu_k, Sigma_k)
            
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            1D array of log-likelihood values. Lower scores indicate anomalous out-of-distribution points.
        """
        if not self.is_fitted or self.gmm is None:
            raise RuntimeError("GMMAnomalyDetector must be fitted before scoring samples.")
        
        # score_samples returns log-likelihood of each sample
        scores = self.gmm.score_samples(X)
        return scores

    def score_single(self, x: np.ndarray) -> float:
        """
        Calculate log-likelihood for a single 1D feature vector.
        
        Args:
            x: 1D array of shape (n_features,).
            
        Returns:
            Scalar float representing log-likelihood.
        """
        x_2d = x.reshape(1, -1)
        score = float(self.score_samples(x_2d)[0])
        return score

    @property
    def means_(self) -> np.ndarray:
        """Return cluster centroids of shape (n_components, n_features)."""
        if not self.is_fitted or self.gmm is None:
            raise RuntimeError("Model is not fitted.")
        return self.gmm.means_

    @property
    def covariances_(self) -> np.ndarray:
        """Return full covariance matrices of shape (n_components, n_features, n_features)."""
        if not self.is_fitted or self.gmm is None:
            raise RuntimeError("Model is not fitted.")
        return self.gmm.covariances_

    @property
    def weights_(self) -> np.ndarray:
        """Return mixture component weights pi_k."""
        if not self.is_fitted or self.gmm is None:
            raise RuntimeError("Model is not fitted.")
        return self.gmm.weights_

    def save(self, filepath: str) -> None:
        """Persist trained GMM detector artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"GMM Anomaly Detector saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "GMMAnomalyDetector":
        """Load persisted GMM detector artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GMM model artifact not found at: {filepath}")
        detector = joblib.load(filepath)
        logger.info(f"Loaded GMM Anomaly Detector from: {filepath}")
        return detector

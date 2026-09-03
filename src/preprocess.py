"""
Feature Engineering and Preprocessing Pipeline for NSL-KDD Network Anomaly Detection.
Implements unsupervised data conditioning, skewness mitigation (log1p), one-hot encoding,
StandardScaler standardization, and PCA dimensionality reduction.
"""

import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger("Preprocessor")

# Canonical 43-column schema of NSL-KDD
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
EXCLUDED_COLS = ["label", "difficulty_level"]
HEAVILY_SKEWED_COLS = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "num_compromised",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "count", "srv_count"
]


class NSLKDDPreprocessor:
    """
    Production-grade preprocessor for NSL-KDD network intrusion telemetry.
    Maintains unsupervised purity by fitting scaling and PCA exclusively on normal network traffic.
    """

    def __init__(self, variance_threshold: float = 0.95, random_state: int = 42):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        
        self.numeric_cols: List[str] = []
        self.log_transformed_cols: List[str] = []
        self.encoded_feature_names: List[str] = []
        self.final_feature_count: int = 0
        self.is_fitted: bool = False

    def _determine_feature_types(self, df: pd.DataFrame) -> None:
        """Categorize DataFrame features into continuous numeric and categorical sets."""
        candidate_numeric = [
            col for col in df.columns
            if col not in CATEGORICAL_COLS and col not in EXCLUDED_COLS
        ]
        self.numeric_cols = candidate_numeric
        self.log_transformed_cols = [
            c for c in HEAVILY_SKEWED_COLS if c in self.numeric_cols
        ]

    def _apply_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply np.log1p(x) transformation to heavily right-skewed continuous attributes.
        Smooths long-tailed outliers to stabilize subsequent Gaussian density estimation.
        """
        df_transformed = df.copy()
        for col in self.log_transformed_cols:
            if col in df_transformed.columns:
                # Ensure non-negative before log1p
                val = np.maximum(0, df_transformed[col].values.astype(np.float64))
                df_transformed[col] = np.log1p(val)
        return df_transformed

    def fit(self, df_train_normal: pd.DataFrame) -> "NSLKDDPreprocessor":
        """
        Fit one-hot encoder, standard scaler, and PCA on normal baseline network records.
        
        Args:
            df_train_normal: DataFrame containing ONLY normal traffic records.
        """
        logger.info(f"Fitting preprocessor on {len(df_train_normal)} normal baseline samples...")
        self._determine_feature_types(df_train_normal)

        # 1. Log Transform
        df_log = self._apply_log_transforms(df_train_normal)

        # 2. Fit OneHotEncoder
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_data = df_log[CATEGORICAL_COLS]
        encoded_cats = self.encoder.fit_transform(cat_data)
        encoded_names = self.encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()

        # Combine continuous + encoded categorical
        num_data = df_log[self.numeric_cols].values.astype(np.float64)
        combined_features = np.hstack([num_data, encoded_cats])
        self.encoded_feature_names = self.numeric_cols + encoded_names

        # 3. Fit StandardScaler
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(combined_features)

        # 4. Fit PCA (>= 95% variance)
        self.pca = PCA(n_components=self.variance_threshold, random_state=self.random_state)
        pca_features = self.pca.fit_transform(scaled_features)
        
        self.final_feature_count = pca_features.shape[1]
        cumulative_variance = np.sum(self.pca.explained_variance_ratio_)
        
        logger.info(
            f"PCA fitted: Retained {self.final_feature_count} components "
            f"explaining {cumulative_variance * 100:.2f}% cumulative variance "
            f"(Original dimension: {combined_features.shape[1]})."
        )
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform arbitrary network records through the pre-fit log-scaling-PCA pipeline.
        
        Args:
            df: Raw DataFrame containing network records.
            
        Returns:
            2D numpy array of shape (n_samples, n_pca_components).
        """
        if not self.is_fitted:
            raise RuntimeError("NSLKDDPreprocessor must be fitted before calling transform().")

        df_log = self._apply_log_transforms(df)
        cat_data = df_log[CATEGORICAL_COLS]
        encoded_cats = self.encoder.transform(cat_data)

        num_data = df_log[self.numeric_cols].values.astype(np.float64)
        combined = np.hstack([num_data, encoded_cats])
        scaled = self.scaler.transform(combined)
        pca_projected = self.pca.transform(scaled)
        return pca_projected

    def fit_transform(self, df_train_normal: pd.DataFrame) -> np.ndarray:
        """Fit on normal records and return projected PCA features."""
        self.fit(df_train_normal)
        return self.transform(df_train_normal)

    def transform_single(self, sample_dict: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single real-time packet feature dictionary into PCA feature space.
        
        Args:
            sample_dict: Dictionary representing single network record.
            
        Returns:
            1D numpy array of shape (n_pca_components,).
        """
        df_single = pd.DataFrame([sample_dict])
        projected = self.transform(df_single)
        return projected.flatten()

    def get_ground_truth_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract binary ground truth from label column.
        0 = Normal, 1 = Attack (Anomaly).
        """
        labels = df["label"].astype(str).str.strip().str.lower()
        # In NSL-KDD, normal records have label == 'normal' or 'normal.'
        y_binary = np.where(labels.str.startswith("normal"), 0, 1)
        return y_binary

    def save(self, filepath: str) -> None:
        """Persist preprocessor state to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "NSLKDDPreprocessor":
        """Load pre-trained preprocessor instance from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor artifact not found at: {filepath}")
        obj = joblib.load(filepath)
        logger.info(f"Loaded preprocessor from: {filepath}")
        return obj

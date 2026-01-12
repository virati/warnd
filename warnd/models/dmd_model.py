"""
DMD-based predictor for depression onset prediction.

This module implements a complete prediction pipeline using Dynamic Mode Decomposition
to capture temporal dynamics from multi-stage longitudinal data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .dmd_core import DMDCore


class DMDPredictor:
    """
    DMD-based predictor for depression onset.
    
    This predictor processes multi-stage data:
    - Stage 1: Baseline questionnaire data
    - Stage 2: EMA surveys and Garmin wearable features (temporal)
    - Stage 3: Quarterly questionnaire data
    
    Uses DMD to extract temporal features from Stage 2 data and combines
    with cross-sectional data from Stages 1 and 3.
    """
    
    def __init__(
        self,
        dmd_rank: Optional[int] = None,
        classifier_type: str = "gradient_boosting",
        random_state: int = 42
    ):
        """
        Initialize DMD predictor.
        
        Args:
            dmd_rank: Rank for DMD truncation (None for full rank)
            classifier_type: Type of classifier ("random_forest" or "gradient_boosting")
            random_state: Random seed for reproducibility
        """
        self.dmd_rank = dmd_rank
        self.random_state = random_state
        
        # Initialize DMD for Stage 2 temporal data
        self.dmd = DMDCore(rank=dmd_rank)
        
        # Initialize scaler for features
        self.scaler = StandardScaler()
        
        # Initialize classifier
        if classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight='balanced'
            )
        elif classifier_type == "gradient_boosting":
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.is_fitted = False
        
    def _extract_dmd_features_stage2(self, stage2_data: np.ndarray) -> np.ndarray:
        """
        Extract DMD features from Stage 2 temporal data.
        
        Args:
            stage2_data: Array of shape (n_samples, n_timesteps, n_features)
            
        Returns:
            DMD features for each sample
        """
        n_samples = stage2_data.shape[0]
        dmd_features_list = []
        
        for i in range(n_samples):
            # Transpose to (n_features, n_timesteps) for DMD
            sample_data = stage2_data[i].T
            
            if sample_data.shape[1] < 2:
                # Not enough time steps, use mean features
                dmd_features = np.concatenate([
                    np.mean(sample_data, axis=1),
                    np.std(sample_data, axis=1),
                    np.zeros(sample_data.shape[0] * 3)  # Placeholder for missing DMD features
                ])
            else:
                # Extract DMD features
                try:
                    dmd_features = self.dmd.extract_features(sample_data)
                except Exception as e:
                    # Fallback to statistical features if DMD fails
                    dmd_features = np.concatenate([
                        np.mean(sample_data, axis=1),
                        np.std(sample_data, axis=1),
                        np.min(sample_data, axis=1),
                        np.max(sample_data, axis=1)
                    ])
            
            dmd_features_list.append(dmd_features)
        
        # Ensure all feature vectors have the same length by padding if necessary
        max_len = max(len(f) for f in dmd_features_list)
        padded_features = []
        for f in dmd_features_list:
            if len(f) < max_len:
                padded = np.pad(f, (0, max_len - len(f)), mode='constant')
            else:
                padded = f
            padded_features.append(padded)
        
        return np.array(padded_features)
    
    def _process_stage1_features(self, stage1_data: np.ndarray) -> np.ndarray:
        """
        Process Stage 1 questionnaire data.
        
        Args:
            stage1_data: Array of shape (n_samples, n_features)
            
        Returns:
            Processed Stage 1 features
        """
        return stage1_data
    
    def _process_stage3_features(self, stage3_data: np.ndarray) -> np.ndarray:
        """
        Process Stage 3 quarterly questionnaire data.
        
        Args:
            stage3_data: Array of shape (n_samples, n_timepoints, n_features)
                         where n_timepoints = 8 (quarterly)
            
        Returns:
            Processed Stage 3 features
        """
        # Extract temporal statistics from quarterly data
        if stage3_data.ndim == 3:
            stage3_mean = np.mean(stage3_data, axis=1)
            stage3_std = np.std(stage3_data, axis=1)
            stage3_trend = np.apply_along_axis(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
                axis=1,
                arr=stage3_data
            )
            
            return np.concatenate([stage3_mean, stage3_std, stage3_trend], axis=1)
        else:
            return stage3_data
    
    def fit(
        self,
        stage1_data: np.ndarray,
        stage2_data: np.ndarray,
        stage3_data: np.ndarray,
        y: np.ndarray
    ) -> 'DMDPredictor':
        """
        Fit the DMD predictor.
        
        Args:
            stage1_data: Stage 1 questionnaire data, shape (n_samples, n_features_1)
            stage2_data: Stage 2 temporal data, shape (n_samples, n_timesteps, n_features_2)
            stage3_data: Stage 3 quarterly data, shape (n_samples, n_timepoints, n_features_3)
            y: Binary outcome labels, shape (n_samples,)
            
        Returns:
            self: Fitted predictor
        """
        # Process each stage
        features_stage1 = self._process_stage1_features(stage1_data)
        features_stage2 = self._extract_dmd_features_stage2(stage2_data)
        features_stage3 = self._process_stage3_features(stage3_data)
        
        # Combine all features
        X_combined = np.concatenate([
            features_stage1,
            features_stage2,
            features_stage3
        ], axis=1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)
        
        # Fit classifier
        self.classifier.fit(X_scaled, y)
        
        self.is_fitted = True
        
        return self
    
    def predict(
        self,
        stage1_data: np.ndarray,
        stage2_data: np.ndarray,
        stage3_data: np.ndarray
    ) -> np.ndarray:
        """
        Predict depression onset.
        
        Args:
            stage1_data: Stage 1 questionnaire data
            stage2_data: Stage 2 temporal data
            stage3_data: Stage 3 quarterly data
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Process each stage
        features_stage1 = self._process_stage1_features(stage1_data)
        features_stage2 = self._extract_dmd_features_stage2(stage2_data)
        features_stage3 = self._process_stage3_features(stage3_data)
        
        # Combine all features
        X_combined = np.concatenate([
            features_stage1,
            features_stage2,
            features_stage3
        ], axis=1)
        
        # Scale features
        X_scaled = self.scaler.transform(X_combined)
        
        # Predict
        predictions = self.classifier.predict(X_scaled)
        
        return predictions
    
    def predict_proba(
        self,
        stage1_data: np.ndarray,
        stage2_data: np.ndarray,
        stage3_data: np.ndarray
    ) -> np.ndarray:
        """
        Predict probability of depression onset.
        
        Args:
            stage1_data: Stage 1 questionnaire data
            stage2_data: Stage 2 temporal data
            stage3_data: Stage 3 quarterly data
            
        Returns:
            Probability predictions of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Process each stage
        features_stage1 = self._process_stage1_features(stage1_data)
        features_stage2 = self._extract_dmd_features_stage2(stage2_data)
        features_stage3 = self._process_stage3_features(stage3_data)
        
        # Combine all features
        X_combined = np.concatenate([
            features_stage1,
            features_stage2,
            features_stage3
        ], axis=1)
        
        # Scale features
        X_scaled = self.scaler.transform(X_combined)
        
        # Predict probabilities
        probabilities = self.classifier.predict_proba(X_scaled)
        
        return probabilities
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the classifier.
        
        Returns:
            Feature importance array if available, None otherwise
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        else:
            return None

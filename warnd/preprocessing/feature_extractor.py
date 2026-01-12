"""
Feature extraction for multi-stage data.

Extracts and engineers features from different data stages to capture
temporal, statistical, and domain-specific patterns relevant to depression prediction.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats, signal


class FeatureExtractor:
    """
    Feature extractor for multi-stage depression prediction data.
    
    Extracts various types of features:
    - Statistical features (mean, std, quantiles)
    - Temporal features (trends, seasonality)
    - Frequency domain features (spectral analysis)
    - Nonlinear features (entropy, complexity)
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract_temporal_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from time series data.
        
        Args:
            data: Time series data of shape (n_timesteps, n_features)
            
        Returns:
            Temporal feature vector
        """
        features = []
        
        n_timesteps, n_features = data.shape
        
        for i in range(n_features):
            series = data[:, i]
            
            # Basic statistics
            features.extend([
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.median(series),
                np.percentile(series, 25),
                np.percentile(series, 75)
            ])
            
            # Temporal dynamics
            if n_timesteps > 1:
                # Linear trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    np.arange(n_timesteps), series
                )
                features.extend([slope, r_value])
                
                # Variability
                diff = np.diff(series)
                features.extend([
                    np.mean(np.abs(diff)),  # Mean absolute change
                    np.std(diff)  # Volatility
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def extract_frequency_features(self, data: np.ndarray, fs: float = 1.0) -> np.ndarray:
        """
        Extract frequency domain features using FFT.
        
        Args:
            data: Time series data of shape (n_timesteps, n_features)
            fs: Sampling frequency
            
        Returns:
            Frequency feature vector
        """
        features = []
        
        n_timesteps, n_features = data.shape
        
        if n_timesteps < 4:
            # Not enough data for frequency analysis
            return np.zeros(n_features * 3)
        
        for i in range(n_features):
            series = data[:, i]
            
            # FFT
            fft_vals = np.fft.fft(series)
            fft_freq = np.fft.fftfreq(n_timesteps, 1/fs)
            
            # Power spectrum
            power = np.abs(fft_vals) ** 2
            
            # Dominant frequency
            dominant_idx = np.argmax(power[1:n_timesteps//2]) + 1
            dominant_freq = np.abs(fft_freq[dominant_idx])
            
            # Spectral centroid
            spectral_centroid = np.sum(fft_freq[:n_timesteps//2] * power[:n_timesteps//2]) / np.sum(power[:n_timesteps//2])
            
            # Spectral entropy
            power_normalized = power[:n_timesteps//2] / np.sum(power[:n_timesteps//2])
            spectral_entropy = -np.sum(power_normalized * np.log2(power_normalized + 1e-10))
            
            features.extend([
                dominant_freq,
                spectral_centroid,
                spectral_entropy
            ])
        
        return np.array(features)
    
    def extract_complexity_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract complexity and nonlinear features.
        
        Args:
            data: Time series data of shape (n_timesteps, n_features)
            
        Returns:
            Complexity feature vector
        """
        features = []
        
        n_timesteps, n_features = data.shape
        
        for i in range(n_features):
            series = data[:, i]
            
            # Sample entropy (approximate)
            features.append(self._sample_entropy(series))
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(series)) != 0)
            features.append(zero_crossings / n_timesteps)
            
            # Autocorrelation at lag 1
            if n_timesteps > 1:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0.0
                features.append(autocorr)
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def _sample_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate sample entropy of a time series.
        
        Args:
            series: Time series
            m: Pattern length
            r: Tolerance (fraction of std)
            
        Returns:
            Sample entropy value
        """
        N = len(series)
        if N < m + 1:
            return 0.0
        
        # Normalize
        series_norm = (series - np.mean(series)) / (np.std(series) + 1e-10)
        r_abs = r * np.std(series_norm)
        
        def _maxdist(xi, xj):
            return np.max(np.abs(xi - xj))
        
        # Count patterns
        def _phi(m):
            patterns = np.array([series_norm[i:i+m] for i in range(N-m)])
            count = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j and _maxdist(patterns[i], patterns[j]) < r_abs:
                        count += 1
            return count / (N - m)
        
        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            
            if phi_m == 0 or phi_m1 == 0:
                return 0.0
            
            return -np.log(phi_m1 / phi_m)
        except:
            return 0.0
    
    def extract_stage1_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from Stage 1 questionnaire data.
        
        Args:
            data: Stage 1 data of shape (n_features,)
            
        Returns:
            Feature vector
        """
        # Stage 1 is already in feature form, just return as-is
        # Could add polynomial features or interactions here
        return data
    
    def extract_stage2_features(
        self, 
        data: np.ndarray,
        include_temporal: bool = True,
        include_frequency: bool = True,
        include_complexity: bool = True
    ) -> np.ndarray:
        """
        Extract comprehensive features from Stage 2 temporal data.
        
        Args:
            data: Stage 2 data of shape (n_timesteps, n_features)
            include_temporal: Include temporal features
            include_frequency: Include frequency features
            include_complexity: Include complexity features
            
        Returns:
            Feature vector
        """
        features_list = []
        
        if include_temporal:
            temporal_features = self.extract_temporal_features(data)
            features_list.append(temporal_features)
        
        if include_frequency:
            frequency_features = self.extract_frequency_features(data)
            features_list.append(frequency_features)
        
        if include_complexity:
            complexity_features = self.extract_complexity_features(data)
            features_list.append(complexity_features)
        
        return np.concatenate(features_list)
    
    def extract_stage3_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from Stage 3 quarterly questionnaire data.
        
        Args:
            data: Stage 3 data of shape (8, n_features)
            
        Returns:
            Feature vector
        """
        # Treat quarterly data as short time series
        features = []
        
        # Temporal statistics
        features.append(np.mean(data, axis=0))
        features.append(np.std(data, axis=0))
        features.append(np.min(data, axis=0))
        features.append(np.max(data, axis=0))
        
        # Trends over quarters
        n_quarters, n_features = data.shape
        trends = []
        for i in range(n_features):
            slope, _, _, _, _ = stats.linregress(np.arange(n_quarters), data[:, i])
            trends.append(slope)
        features.append(np.array(trends))
        
        return np.concatenate([f.flatten() for f in features])

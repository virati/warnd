"""
Dynamic Mode Decomposition (DMD) Core Implementation

This module implements the core DMD algorithm for extracting temporal dynamics
from time-series data. DMD is used to identify spatiotemporal coherent structures
and their evolution in complex systems.
"""

import numpy as np
from typing import Optional, Tuple


class DMDCore:
    """
    Core Dynamic Mode Decomposition implementation.
    
    DMD decomposes a time-series dataset into spatial modes and their
    corresponding temporal dynamics, useful for capturing patterns in
    longitudinal health data.
    
    Attributes:
        modes: DMD spatial modes
        eigenvalues: DMD eigenvalues
        amplitudes: DMD mode amplitudes
        omega: Continuous-time eigenvalues
        rank: Truncation rank for SVD
    """
    
    def __init__(self, rank: Optional[int] = None, svd_rank: Optional[int] = None):
        """
        Initialize DMD core.
        
        Args:
            rank: Rank for truncation (default: None for full rank)
            svd_rank: Rank for SVD truncation (default: None for full rank)
        """
        self.rank = rank
        self.svd_rank = svd_rank if svd_rank is not None else rank
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None
        self.omega = None
        self.dynamics = None
        
    def fit(self, X: np.ndarray, dt: float = 1.0) -> 'DMDCore':
        """
        Fit DMD model to data.
        
        Args:
            X: Data matrix of shape (n_features, n_timesteps)
            dt: Time step between snapshots
            
        Returns:
            self: Fitted DMD model
        """
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 time snapshots for DMD")
            
        # Split data into X and Y (shifted by one time step)
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        
        # SVD of X1
        U, s, Vt = np.linalg.svd(X1, full_matrices=False)
        
        # Truncate to rank if specified
        if self.svd_rank is not None and self.svd_rank < len(s):
            U = U[:, :self.svd_rank]
            s = s[:self.svd_rank]
            Vt = Vt[:self.svd_rank, :]
        
        # Build Atilde
        Atilde = U.conj().T @ X2 @ Vt.conj().T @ np.diag(1.0 / s)
        
        # Eigendecomposition of Atilde
        eigenvalues, W = np.linalg.eig(Atilde)
        
        # Compute DMD modes
        self.modes = X2 @ Vt.conj().T @ np.diag(1.0 / s) @ W
        self.eigenvalues = eigenvalues
        
        # Compute continuous-time eigenvalues
        self.omega = np.log(eigenvalues) / dt
        
        # Compute amplitudes using least squares
        self.amplitudes = np.linalg.lstsq(self.modes, X[:, 0], rcond=None)[0]
        
        return self
    
    def reconstruct(self, n_timesteps: int, t0: int = 0) -> np.ndarray:
        """
        Reconstruct data using DMD modes.
        
        Args:
            n_timesteps: Number of time steps to reconstruct
            t0: Starting time index
            
        Returns:
            Reconstructed data matrix
        """
        if self.modes is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        time_indices = np.arange(t0, t0 + n_timesteps)
        
        # Compute dynamics: exp(omega * t)
        dynamics = np.exp(np.outer(self.omega, time_indices))
        
        # Reconstruct: Phi * diag(b) * exp(omega * t)
        reconstruction = self.modes @ np.diag(self.amplitudes) @ dynamics
        
        return np.real(reconstruction)
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        """
        Predict future states using DMD.
        
        Args:
            n_steps: Number of steps to predict into the future
            
        Returns:
            Predicted states
        """
        if self.modes is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Predict by extrapolating the dynamics
        future_dynamics = np.exp(np.outer(self.omega, np.arange(n_steps)))
        prediction = self.modes @ np.diag(self.amplitudes) @ future_dynamics
        
        return np.real(prediction)
    
    def get_mode_importance(self) -> np.ndarray:
        """
        Compute importance of each mode based on amplitude and eigenvalue.
        
        Returns:
            Importance score for each mode
        """
        if self.modes is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Importance based on amplitude magnitude and eigenvalue proximity to unit circle
        importance = np.abs(self.amplitudes) * (1.0 / (1.0 + np.abs(np.abs(self.eigenvalues) - 1.0)))
        
        return importance
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract DMD-based features from data.
        
        Args:
            X: Data matrix of shape (n_features, n_timesteps)
            
        Returns:
            Feature vector combining mode information
        """
        self.fit(X)
        
        # Extract features: mode importance, dominant frequencies, etc.
        mode_importance = self.get_mode_importance()
        
        # Dominant frequencies (imaginary part of omega)
        frequencies = np.imag(self.omega)
        
        # Growth rates (real part of omega)
        growth_rates = np.real(self.omega)
        
        # Combine into feature vector
        features = np.concatenate([
            mode_importance,
            np.abs(frequencies),
            growth_rates,
            np.abs(self.amplitudes),
            np.abs(self.eigenvalues)
        ])
        
        return features

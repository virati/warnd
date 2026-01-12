"""
Data handler for multi-stage depression prediction data.

Handles loading, preprocessing, and structuring data from the three stages:
- Stage 1: Baseline questionnaire
- Stage 2: EMA surveys and Garmin wearable features
- Stage 3: Quarterly questionnaires
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


# Constants for synthetic data generation
SYNTHETIC_FREQ_MIN = 0.1  # Minimum frequency for synthetic temporal signals
SYNTHETIC_FREQ_MAX = 2.0  # Maximum frequency for synthetic temporal signals
SYNTHETIC_TREND_MIN = -0.1  # Minimum trend value for synthetic data
SYNTHETIC_TREND_MAX = 0.1  # Maximum trend value for synthetic data
SYNTHETIC_NOISE_STD = 0.3  # Standard deviation of noise in synthetic data


class DataHandler:
    """
    Handler for multi-stage longitudinal data.
    
    Manages data from three distinct stages with different temporal resolutions
    and combines them for model training and prediction.
    """
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize data handler.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.stage1_data = None
        self.stage2_data = None
        self.stage3_data = None
        self.outcomes = None
        self.participant_ids = None
        
    def load_data(
        self,
        stage1_path: Optional[Union[str, Path]] = None,
        stage2_path: Optional[Union[str, Path]] = None,
        stage3_path: Optional[Union[str, Path]] = None,
        outcomes_path: Optional[Union[str, Path]] = None
    ) -> 'DataHandler':
        """
        Load data from files.
        
        Args:
            stage1_path: Path to Stage 1 data file
            stage2_path: Path to Stage 2 data file
            stage3_path: Path to Stage 3 data file
            outcomes_path: Path to outcomes file
            
        Returns:
            self: DataHandler with loaded data
        """
        if stage1_path:
            self.stage1_data = self._load_stage1(stage1_path)
        
        if stage2_path:
            self.stage2_data = self._load_stage2(stage2_path)
        
        if stage3_path:
            self.stage3_data = self._load_stage3(stage3_path)
        
        if outcomes_path:
            self.outcomes = self._load_outcomes(outcomes_path)
        
        return self
    
    def _load_stage1(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load Stage 1 baseline questionnaire data.
        
        Expected format: One row per participant with questionnaire responses.
        
        Args:
            path: Path to Stage 1 data file
            
        Returns:
            DataFrame with Stage 1 data
        """
        df = pd.read_csv(path)
        return df
    
    def _load_stage2(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load Stage 2 temporal data (EMA + Garmin).
        
        Expected format: 352-356 rows per participant with temporal features.
        
        Args:
            path: Path to Stage 2 data file
            
        Returns:
            DataFrame with Stage 2 data
        """
        df = pd.read_csv(path)
        return df
    
    def _load_stage3(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load Stage 3 quarterly questionnaire data.
        
        Expected format: 8 quarterly timepoints per participant.
        
        Args:
            path: Path to Stage 3 data file
            
        Returns:
            DataFrame with Stage 3 data
        """
        df = pd.read_csv(path)
        return df
    
    def _load_outcomes(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load outcome labels.
        
        Expected format: Binary depression onset indicator at Stage 3 timepoints.
        
        Args:
            path: Path to outcomes file
            
        Returns:
            DataFrame with outcome labels
        """
        df = pd.read_csv(path)
        return df
    
    def prepare_data(
        self,
        participant_id_col: str = 'participant_id',
        timepoint_col: str = 'timepoint'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Structures the multi-stage data into arrays suitable for the DMD predictor.
        
        Args:
            participant_id_col: Column name for participant IDs
            timepoint_col: Column name for timepoints
            
        Returns:
            Tuple of (stage1_array, stage2_array, stage3_array, outcomes_array)
        """
        if self.stage1_data is None or self.stage2_data is None or self.stage3_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get unique participant IDs
        participants = self._get_common_participants(
            self.stage1_data, self.stage2_data, self.stage3_data,
            participant_id_col
        )
        
        self.participant_ids = participants
        
        # Prepare Stage 1 data (cross-sectional)
        stage1_array = self._prepare_stage1_array(
            self.stage1_data, participants, participant_id_col
        )
        
        # Prepare Stage 2 data (temporal)
        stage2_array = self._prepare_stage2_array(
            self.stage2_data, participants, participant_id_col, timepoint_col
        )
        
        # Prepare Stage 3 data (quarterly)
        stage3_array = self._prepare_stage3_array(
            self.stage3_data, participants, participant_id_col, timepoint_col
        )
        
        # Prepare outcomes
        if self.outcomes is not None:
            outcomes_array = self._prepare_outcomes_array(
                self.outcomes, participants, participant_id_col
            )
        else:
            outcomes_array = None
        
        return stage1_array, stage2_array, stage3_array, outcomes_array
    
    def _get_common_participants(
        self,
        *dataframes: pd.DataFrame,
        participant_id_col: str = 'participant_id'
    ) -> List:
        """
        Get list of participants common to all dataframes.
        
        Args:
            dataframes: DataFrames to find common participants in
            participant_id_col: Column name for participant IDs
            
        Returns:
            List of common participant IDs
        """
        participant_sets = [set(df[participant_id_col].unique()) for df in dataframes]
        common_participants = set.intersection(*participant_sets)
        return sorted(list(common_participants))
    
    def _prepare_stage1_array(
        self,
        df: pd.DataFrame,
        participants: List,
        participant_id_col: str
    ) -> np.ndarray:
        """
        Prepare Stage 1 data as array.
        
        Args:
            df: Stage 1 DataFrame
            participants: List of participant IDs
            participant_id_col: Column name for participant IDs
            
        Returns:
            Array of shape (n_participants, n_features)
        """
        # Remove ID column
        feature_cols = [col for col in df.columns if col != participant_id_col]
        
        # Sort by participants and extract features
        df_sorted = df[df[participant_id_col].isin(participants)].copy()
        df_sorted = df_sorted.sort_values(participant_id_col)
        
        return df_sorted[feature_cols].values
    
    def _prepare_stage2_array(
        self,
        df: pd.DataFrame,
        participants: List,
        participant_id_col: str,
        timepoint_col: str
    ) -> np.ndarray:
        """
        Prepare Stage 2 data as 3D array.
        
        Args:
            df: Stage 2 DataFrame
            participants: List of participant IDs
            participant_id_col: Column name for participant IDs
            timepoint_col: Column name for timepoints
            
        Returns:
            Array of shape (n_participants, n_timesteps, n_features)
        """
        # Remove ID and timepoint columns
        feature_cols = [
            col for col in df.columns 
            if col not in [participant_id_col, timepoint_col]
        ]
        
        # Filter to participants
        df_filtered = df[df[participant_id_col].isin(participants)].copy()
        
        # Group by participant and create 3D array
        arrays = []
        for pid in participants:
            participant_data = df_filtered[df_filtered[participant_id_col] == pid].copy()
            participant_data = participant_data.sort_values(timepoint_col)
            arrays.append(participant_data[feature_cols].values)
        
        # Pad sequences to same length
        max_len = max(arr.shape[0] for arr in arrays)
        padded_arrays = []
        for arr in arrays:
            if arr.shape[0] < max_len:
                padding = np.zeros((max_len - arr.shape[0], arr.shape[1]))
                arr_padded = np.vstack([arr, padding])
            else:
                arr_padded = arr
            padded_arrays.append(arr_padded)
        
        return np.array(padded_arrays)
    
    def _prepare_stage3_array(
        self,
        df: pd.DataFrame,
        participants: List,
        participant_id_col: str,
        timepoint_col: str
    ) -> np.ndarray:
        """
        Prepare Stage 3 data as 3D array.
        
        Args:
            df: Stage 3 DataFrame
            participants: List of participant IDs
            participant_id_col: Column name for participant IDs
            timepoint_col: Column name for timepoints
            
        Returns:
            Array of shape (n_participants, 8, n_features)
        """
        # Remove ID and timepoint columns
        feature_cols = [
            col for col in df.columns 
            if col not in [participant_id_col, timepoint_col]
        ]
        
        # Filter to participants
        df_filtered = df[df[participant_id_col].isin(participants)].copy()
        
        # Group by participant and create 3D array
        arrays = []
        for pid in participants:
            participant_data = df_filtered[df_filtered[participant_id_col] == pid].copy()
            participant_data = participant_data.sort_values(timepoint_col)
            arrays.append(participant_data[feature_cols].values)
        
        # Ensure all have 8 timepoints
        expected_timepoints = 8
        padded_arrays = []
        for arr in arrays:
            if arr.shape[0] < expected_timepoints:
                padding = np.zeros((expected_timepoints - arr.shape[0], arr.shape[1]))
                arr_padded = np.vstack([arr, padding])
            elif arr.shape[0] > expected_timepoints:
                arr_padded = arr[:expected_timepoints]
            else:
                arr_padded = arr
            padded_arrays.append(arr_padded)
        
        return np.array(padded_arrays)
    
    def _prepare_outcomes_array(
        self,
        df: pd.DataFrame,
        participants: List,
        participant_id_col: str
    ) -> np.ndarray:
        """
        Prepare outcomes as array.
        
        Args:
            df: Outcomes DataFrame
            participants: List of participant IDs
            participant_id_col: Column name for participant IDs
            
        Returns:
            Binary outcome array of shape (n_participants,)
        """
        df_sorted = df[df[participant_id_col].isin(participants)].copy()
        df_sorted = df_sorted.sort_values(participant_id_col)
        
        # Assuming outcome column is named 'outcome' or 'depression_onset'
        outcome_col = None
        for col in ['outcome', 'depression_onset', 'label', 'target']:
            if col in df_sorted.columns:
                outcome_col = col
                break
        
        if outcome_col is None:
            # Use first non-ID column as outcome
            outcome_col = [col for col in df_sorted.columns if col != participant_id_col][0]
        
        return df_sorted[outcome_col].values
    
    def create_synthetic_data(
        self,
        n_participants: int = 100,
        n_stage2_timesteps: int = 354,
        n_stage1_features: int = 20,
        n_stage2_features: int = 30,
        n_stage3_features: int = 15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create synthetic data for testing.
        
        Args:
            n_participants: Number of participants
            n_stage2_timesteps: Number of Stage 2 timesteps (352-356)
            n_stage1_features: Number of Stage 1 features
            n_stage2_features: Number of Stage 2 features
            n_stage3_features: Number of Stage 3 features
            random_state: Random seed
            
        Returns:
            Tuple of (stage1_array, stage2_array, stage3_array, outcomes_array)
        """
        np.random.seed(random_state)
        
        # Stage 1: Cross-sectional baseline
        stage1_data = np.random.randn(n_participants, n_stage1_features)
        
        # Stage 2: Temporal data with some structure
        stage2_data = np.zeros((n_participants, n_stage2_timesteps, n_stage2_features))
        for i in range(n_participants):
            # Add temporal dynamics
            t = np.linspace(0, 10, n_stage2_timesteps)
            for j in range(n_stage2_features):
                freq = np.random.uniform(SYNTHETIC_FREQ_MIN, SYNTHETIC_FREQ_MAX)
                phase = np.random.uniform(0, 2*np.pi)
                trend = np.random.uniform(SYNTHETIC_TREND_MIN, SYNTHETIC_TREND_MAX)
                signal = np.sin(freq * t + phase) + trend * t
                noise = np.random.randn(n_stage2_timesteps) * SYNTHETIC_NOISE_STD
                stage2_data[i, :, j] = signal + noise
        
        # Stage 3: Quarterly data (8 timepoints)
        stage3_data = np.random.randn(n_participants, 8, n_stage3_features)
        
        # Outcomes: Binary depression onset
        # Create some correlation with data
        risk_score = np.mean(stage1_data, axis=1) + np.mean(stage2_data[:, -50:, :], axis=(1, 2))
        outcomes = (risk_score > np.median(risk_score)).astype(int)
        
        return stage1_data, stage2_data, stage3_data, outcomes

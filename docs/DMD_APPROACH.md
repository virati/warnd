# DMD-Based Approach for Depression Prediction

## Overview

This document describes the Dynamic Mode Decomposition (DMD) based approach implemented for depression onset prediction using multi-stage longitudinal data.

## Problem Description

The challenge involves predicting depression onset using data from three distinct stages:

- **Stage 1**: Baseline questionnaire (1 row per participant)
- **Stage 2**: EMA surveys and Garmin wearable features (352-356 temporal measurements)
- **Stage 3**: Quarterly questionnaires (8 timepoints)

**Outcome**: Binary depression onset indicator at Stage 3 timepoints

**Evaluation**: Models are ranked by:
1. **Primary**: AUC-ROC (Area Under the ROC Curve)
2. **Tie-breaker 1**: Log Loss
3. **Tie-breaker 2**: F1 Score

## Why Dynamic Mode Decomposition?

DMD is a data-driven method for analyzing complex dynamical systems that offers several advantages for this problem:

### 1. Temporal Dynamics Extraction
- DMD decomposes time-series data into spatial modes and their temporal evolution
- Captures coherent spatiotemporal patterns in longitudinal health data
- Identifies dominant frequencies and growth/decay rates

### 2. Dimensionality Reduction
- Reduces high-dimensional temporal data (352-356 × 30 features) to a manageable set of features
- Extracts the most informative modes using rank truncation
- Provides interpretable features based on physical dynamics

### 3. Handling Irregular Temporal Data
- Works with varying lengths of temporal sequences
- Robust to missing data and noise
- Captures both periodic and aperiodic patterns

### 4. Prediction Capabilities
- DMD modes can be used to forecast future states
- Growth rates indicate stability or instability of patterns
- Eigenvalues near the unit circle indicate persistent behaviors

## Implementation Details

### DMD Core Algorithm

The DMD algorithm operates on a data matrix **X** with snapshots arranged as columns:

```
X = [x₁ x₂ x₃ ... xₙ]
```

1. **Split data**: Create X₁ = [x₁...xₙ₋₁] and X₂ = [x₂...xₙ]
2. **SVD**: Compute X₁ = UΣV*
3. **Reduced operator**: Ã = U* X₂ V Σ⁻¹
4. **Eigendecomposition**: Ã = WΛW⁻¹
5. **DMD modes**: Φ = X₂ V Σ⁻¹ W

**Key outputs**:
- **Modes (Φ)**: Spatial patterns in the data
- **Eigenvalues (λ)**: Growth/decay rates and frequencies
- **Amplitudes (b)**: Importance of each mode

### Feature Extraction from DMD

From the DMD decomposition, we extract several types of features:

1. **Mode Importance**: `|bᵢ| / (1 + ||λᵢ| - 1|)`
   - Combines amplitude and proximity to unit circle
   - Higher values indicate more important modes

2. **Dominant Frequencies**: `Im(ωᵢ)` where `ωᵢ = log(λᵢ)/Δt`
   - Identifies periodic patterns in the data
   - Useful for detecting circadian rhythms or weekly patterns

3. **Growth Rates**: `Re(ωᵢ)`
   - Positive values indicate growing patterns
   - Negative values indicate decaying patterns
   - Important for identifying deteriorating mental health

4. **Eigenvalue Magnitudes**: `|λᵢ|`
   - Values > 1 indicate unstable dynamics
   - Values < 1 indicate stable dynamics

### Multi-Stage Integration

The model combines features from all three stages:

```python
Features = [
    Stage1_features,           # Baseline questionnaire
    DMD_features(Stage2),      # Temporal dynamics from EMA/wearables
    Temporal_stats(Stage3)     # Quarterly trends
]
```

**Stage 1 Processing**:
- Direct use of questionnaire responses
- No temporal processing needed

**Stage 2 Processing**:
1. Apply DMD to each participant's temporal data
2. Extract mode-based features
3. Combine with statistical features (mean, std, trends)

**Stage 3 Processing**:
1. Compute temporal statistics across quarters
2. Extract linear trends
3. Measure variability

### Classification

After feature extraction, we use ensemble methods:

**Gradient Boosting** (default):
- Handles complex non-linear relationships
- Robust to feature scaling
- Provides feature importance
- Better generalization for small datasets

**Random Forest** (alternative):
- Ensemble of decision trees
- Reduces overfitting
- Handles mixed feature types well

Both classifiers use:
- Class weighting for imbalanced data
- Cross-validation for hyperparameter tuning
- Probability calibration for better AUC-ROC

## Advantages of This Approach

1. **Interpretability**: DMD modes have physical meaning
   - Can identify specific temporal patterns associated with depression
   - Feature importance reveals which dynamics are predictive

2. **Efficiency**: DMD provides compact representation
   - Reduces 10,620 temporal features (354 × 30) to ~50-100 DMD features
   - Faster training and prediction

3. **Robustness**: Handles noise and missing data
   - SVD-based approach is stable
   - Rank truncation removes noise

4. **Domain Appropriate**: Well-suited for physiological/behavioral data
   - Mental health has temporal dynamics
   - Sleep, activity, mood have patterns DMD can capture
   - Longitudinal nature of data matches DMD assumptions

## Potential Extensions

1. **Time-delay Embedding**: For better capturing dependencies
2. **Extended DMD**: Include control inputs or external factors
3. **Multi-resolution DMD**: Capture patterns at different timescales
4. **Online DMD**: Update models as new data arrives
5. **Optimal DMD**: Better for noisy data

## References

### Core DMD Papers
- Schmid, P. J. (2010). "Dynamic mode decomposition of numerical and experimental data." *Journal of Fluid Mechanics*, 656, 5-28.
- Kutz, J. N., et al. (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM.

### DMD in Health/Neuroscience
- Brunton, B. W., et al. (2016). "Extracting spatial-temporal coherent patterns in large-scale neural recordings using dynamic mode decomposition." *Journal of Neuroscience Methods*, 258, 1-15.
- Proctor, J. L., & Eckhoff, P. A. (2015). "Discovering dynamic patterns from infectious disease data using dynamic mode decomposition." *International Health*, 7(2), 139-145.

### Depression Prediction
- Jacobson, N. C., & Chung, Y. J. (2020). "Passive sensing of prediction of moment-to-moment depressed mood among undergraduates with clinical levels of depression sample using smartphones." *Sensors*, 20(12), 3572.

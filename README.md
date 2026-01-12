# WARND: DMD-Based Depression Prediction

A Dynamic Mode Decomposition (DMD) based approach for predicting depression onset using multi-stage longitudinal data.

## Overview

This repository implements a machine learning pipeline that uses Dynamic Mode Decomposition to extract temporal dynamics from multi-stage health data for depression prediction. The approach is designed for datasets with three predictor stages:

- **Stage 1**: Baseline questionnaire (1 row per participant)
- **Stage 2**: EMA surveys and Garmin wearable features (352-356 temporal rows per participant)
- **Stage 3**: Quarterly questionnaires (8 timepoints per participant)

The model predicts binary depression onset at Stage 3 timepoints.

## Key Features

### Dynamic Mode Decomposition (DMD)
- Extracts temporal dynamics from Stage 2 time-series data
- Captures spatiotemporal coherent structures in longitudinal health measurements
- Provides interpretable features: mode importance, dominant frequencies, growth rates

### Multi-Stage Integration
- Combines cross-sectional and temporal data from three stages
- Statistical feature extraction from questionnaires
- Temporal feature extraction from wearables and EMA data

### Comprehensive Evaluation
- Primary metric: **AUC-ROC**
- Tie-breaker 1: **Log Loss**
- Tie-breaker 2: **F1 Score**
- Additional metrics: Precision, Recall, Specificity, Confusion Matrix

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/virati/warnd.git
cd warnd

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Example with Synthetic Data

Run the example script to see the model in action with synthetic data:

```bash
python example.py
```

This will:
1. Generate synthetic multi-stage longitudinal data
2. Train a DMD-based predictor
3. Evaluate the model using competition metrics
4. Display feature importance

### Training a Model

Train a model on your own data:

```bash
python train.py --data-dir /path/to/data --output-dir ./output --cv
```

Options:
- `--data-dir`: Directory containing stage1.csv, stage2.csv, stage3.csv, outcomes.csv
- `--output-dir`: Directory to save trained model and results
- `--dmd-rank`: DMD truncation rank (default: None for full rank)
- `--classifier`: Classifier type ('gradient_boosting' or 'random_forest')
- `--cv`: Run cross-validation
- `--n-folds`: Number of CV folds (default: 5)

### Making Predictions

Use a trained model to make predictions:

```bash
python predict.py --model output/dmd_model.pkl --data-dir /path/to/new/data --output predictions.csv
```

## Usage Example (Python API)

```python
import numpy as np
from warnd.models.dmd_model import DMDPredictor
from warnd.preprocessing.data_handler import DataHandler
from warnd.evaluation.metrics import evaluate_model

# Create synthetic data for demonstration
data_handler = DataHandler()
stage1, stage2, stage3, outcomes = data_handler.create_synthetic_data(
    n_participants=100,
    n_stage2_timesteps=354
)

# Split data (use your preferred method)
# ... train/test split code ...

# Initialize and train model
model = DMDPredictor(
    dmd_rank=10,
    classifier_type='gradient_boosting'
)
model.fit(stage1_train, stage2_train, stage3_train, y_train)

# Make predictions
predictions = model.predict(stage1_test, stage2_test, stage3_test)
probabilities = model.predict_proba(stage1_test, stage2_test, stage3_test)

# Evaluate
metrics = evaluate_model(y_test, predictions, probabilities[:, 1])
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
```

## Data Format

### Expected Input Files

#### stage1.csv
Baseline questionnaire with one row per participant:
```
participant_id,feature1,feature2,...,featureN
P001,1.2,3.4,...,5.6
P002,2.3,4.5,...,6.7
```

#### stage2.csv
Temporal data with 352-356 rows per participant:
```
participant_id,timepoint,feature1,feature2,...,featureN
P001,1,1.2,3.4,...,5.6
P001,2,1.3,3.5,...,5.7
...
```

#### stage3.csv
Quarterly data with 8 rows per participant:
```
participant_id,timepoint,feature1,feature2,...,featureN
P001,1,1.2,3.4,...,5.6
P001,2,1.3,3.5,...,5.7
...
P001,8,1.4,3.6,...,5.8
```

#### outcomes.csv
Binary depression onset labels:
```
participant_id,depression_onset
P001,0
P002,1
```

## Architecture

### DMD Core
- Implements Dynamic Mode Decomposition algorithm
- Extracts temporal modes, eigenvalues, and dynamics
- Computes mode importance and frequency features

### DMD Predictor
- Processes multi-stage data
- Extracts DMD features from Stage 2 temporal data
- Combines features from all stages
- Uses ensemble classifier (Gradient Boosting or Random Forest)

### Data Handler
- Loads and structures multi-stage data
- Handles variable-length temporal sequences
- Supports both real and synthetic data

### Feature Extractor
- Temporal features: trends, variability, statistics
- Frequency features: spectral analysis, dominant frequencies
- Complexity features: entropy, autocorrelation

### Evaluation Metrics
- Competition metrics: AUC-ROC, Log Loss, F1 Score
- Additional metrics: Precision, Recall, Specificity
- Cross-validation support

## Model Performance

The model is evaluated using:

1. **AUC-ROC** (Primary): Area under ROC curve
2. **Log Loss** (Tie-breaker 1): Cross-entropy loss
3. **F1 Score** (Tie-breaker 2): Harmonic mean of precision and recall

## Project Structure

```
warnd/
├── warnd/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dmd_core.py          # Core DMD implementation
│   │   └── dmd_model.py         # DMD-based predictor
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_handler.py      # Data loading and preprocessing
│   │   └── feature_extractor.py # Feature engineering
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging utilities
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── example.py                    # Example usage
├── requirements.txt
├── setup.py
└── README.md
```

## References

### Dynamic Mode Decomposition
- Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM.
- Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. *Journal of Fluid Mechanics*, 656, 5-28.

### Applications in Health Data
- DMD has been successfully applied to analyze temporal patterns in physiological signals, EEG data, and longitudinal health measurements.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{warnd2026,
  title={WARND: DMD-Based Depression Prediction},
  author={WARND Team},
  year={2026},
  url={https://github.com/virati/warnd}
}
```

# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/virati/warnd.git
cd warnd

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Running the Example

The fastest way to see the model in action is to run the example script with synthetic data:

```bash
python example.py
```

This will:
1. Generate synthetic multi-stage longitudinal data (200 participants)
2. Split data into train/test sets
3. Train a DMD-based predictor
4. Evaluate using competition metrics (AUC-ROC, Log Loss, F1 Score)
5. Display feature importance

## Training on Your Data

### 1. Prepare Your Data

Organize your data in the following CSV format:

**stage1.csv** (Baseline questionnaire):
```csv
participant_id,feature1,feature2,...,featureN
P001,1.2,3.4,...,5.6
P002,2.3,4.5,...,6.7
```

**stage2.csv** (EMA surveys and wearables, 352-356 rows per participant):
```csv
participant_id,timepoint,feature1,feature2,...,featureN
P001,1,1.2,3.4,...,5.6
P001,2,1.3,3.5,...,5.7
```

**stage3.csv** (Quarterly questionnaires, 8 rows per participant):
```csv
participant_id,timepoint,feature1,feature2,...,featureN
P001,1,1.2,3.4,...,5.6
P001,2,1.3,3.5,...,5.7
...
P001,8,1.4,3.6,...,5.8
```

**outcomes.csv** (Binary depression onset):
```csv
participant_id,depression_onset
P001,0
P002,1
```

### 2. Train the Model

```bash
python train.py \
  --data-dir ./data \
  --output-dir ./output \
  --dmd-rank 10 \
  --classifier gradient_boosting \
  --cv \
  --n-folds 5
```

**Parameters**:
- `--data-dir`: Directory containing your CSV files
- `--output-dir`: Where to save the trained model and results
- `--dmd-rank`: DMD truncation rank (lower = more dimensionality reduction)
- `--classifier`: 'gradient_boosting' or 'random_forest'
- `--cv`: Enable cross-validation
- `--n-folds`: Number of CV folds (default: 5)

### 3. Make Predictions

```bash
python predict.py \
  --model output/dmd_model.pkl \
  --data-dir ./new_data \
  --output predictions.csv
```

## Using the Python API

### Basic Usage

```python
from warnd.models.dmd_model import DMDPredictor
from warnd.preprocessing.data_handler import DataHandler
from warnd.evaluation.metrics import evaluate_model

# Load or create data
data_handler = DataHandler()
stage1, stage2, stage3, outcomes = data_handler.create_synthetic_data(
    n_participants=100
)

# Split data (train/test)
from sklearn.model_selection import train_test_split
# ... your splitting code ...

# Train model
model = DMDPredictor(dmd_rank=10, classifier_type='gradient_boosting')
model.fit(stage1_train, stage2_train, stage3_train, y_train)

# Predict
predictions = model.predict(stage1_test, stage2_test, stage3_test)
probabilities = model.predict_proba(stage1_test, stage2_test, stage3_test)

# Evaluate
metrics = evaluate_model(y_test, predictions, probabilities[:, 1])
```

### Advanced Configuration

```python
from warnd.utils.config import Config

# Create custom configuration
config = Config({
    'dmd_rank': 15,
    'classifier_type': 'random_forest',
    'test_size': 0.2,
    'random_state': 123
})

# Save configuration
config.save('my_config.json')

# Load configuration
config = Config.load('my_config.json')
```

## Understanding the Results

After training, you'll see metrics like:

```
Competition Metrics (Test Set):
  🎯 Primary Metric - AUC-ROC:     0.8234
  🔹 Tie-breaker 1 - Log Loss:    0.4567
  🔹 Tie-breaker 2 - F1 Score:    0.7654
```

**Interpreting the metrics**:

- **AUC-ROC** (0-1, higher is better): Measures the model's ability to distinguish between classes
  - 0.5 = random guessing
  - 0.7-0.8 = acceptable
  - 0.8-0.9 = excellent
  - 0.9+ = outstanding

- **Log Loss** (0+, lower is better): Measures the accuracy of probability predictions
  - 0 = perfect predictions
  - ~0.69 = random guessing for binary classification

- **F1 Score** (0-1, higher is better): Harmonic mean of precision and recall
  - Good for imbalanced datasets
  - Balances false positives and false negatives

## Tuning the Model

### DMD Rank
- Lower rank (5-10): More dimensionality reduction, faster, may lose information
- Higher rank (20-30): Preserves more temporal dynamics, slower, may overfit
- None: Use full rank (all modes)

### Classifier Choice
- **Gradient Boosting**: Usually better for tabular data, more prone to overfitting
- **Random Forest**: More robust, good for noisy data, easier to tune

### Tips for Better Performance
1. **Feature engineering**: Add domain-specific features from questionnaires
2. **Hyperparameter tuning**: Use grid search or random search
3. **Cross-validation**: Always validate on held-out data
4. **Class balancing**: Use SMOTE or class weights for imbalanced data
5. **Ensemble**: Combine multiple models for better predictions

## Troubleshooting

### "Not enough time steps for DMD"
- Ensure Stage 2 data has at least 2 time points per participant
- Check for missing data in temporal sequences

### "Memory error"
- Reduce `dmd_rank` to use less memory
- Process data in batches
- Use a machine with more RAM

### "Poor model performance"
- Try different `dmd_rank` values (5, 10, 15, 20)
- Experiment with both classifiers
- Check for data quality issues
- Ensure proper train/test split (no data leakage)
- Consider feature normalization

## Running Tests

```bash
python -m unittest tests.test_warnd -v
```

All 22 tests should pass.

## Next Steps

1. Read the [full documentation](docs/DMD_APPROACH.md) to understand the methodology
2. Experiment with different hyperparameters
3. Add your domain-specific features
4. Implement custom feature extractors if needed
5. Try ensemble methods for better performance

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- See [docs/DMD_APPROACH.md](docs/DMD_APPROACH.md) for theoretical background
- Review test cases in `tests/test_warnd.py` for usage examples

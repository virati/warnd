"""
Example usage of the WARND DMD-based depression prediction model.

This script demonstrates how to use the model with synthetic data.
"""

import numpy as np
from warnd.models.dmd_model import DMDPredictor
from warnd.preprocessing.data_handler import DataHandler
from warnd.evaluation.metrics import evaluate_model
from warnd.utils.config import Config
from warnd.utils.logger import setup_logger

from sklearn.model_selection import train_test_split


def main():
    """Run example with synthetic data."""
    
    # Set up logger
    logger = setup_logger()
    logger.info("=" * 70)
    logger.info("WARND: DMD-Based Depression Prediction - Example Usage")
    logger.info("=" * 70)
    
    # Create configuration
    config = Config({
        'n_participants': 200,
        'stage1_features': 20,
        'stage2_features': 30,
        'stage2_timesteps': 354,
        'stage3_features': 15,
        'dmd_rank': 10,
        'classifier_type': 'gradient_boosting',
        'test_size': 0.25,
        'random_state': 42
    })
    
    logger.info("\nConfiguration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Create synthetic data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Creating Synthetic Data")
    logger.info("=" * 70)
    logger.info("Generating multi-stage longitudinal data...")
    logger.info(f"  - Stage 1: {config['n_participants']} participants × {config['stage1_features']} features")
    logger.info(f"  - Stage 2: {config['n_participants']} participants × {config['stage2_timesteps']} timesteps × {config['stage2_features']} features")
    logger.info(f"  - Stage 3: {config['n_participants']} participants × 8 quarters × {config['stage3_features']} features")
    
    data_handler = DataHandler()
    stage1_data, stage2_data, stage3_data, outcomes = data_handler.create_synthetic_data(
        n_participants=config['n_participants'],
        n_stage2_timesteps=config['stage2_timesteps'],
        n_stage1_features=config['stage1_features'],
        n_stage2_features=config['stage2_features'],
        n_stage3_features=config['stage3_features'],
        random_state=config['random_state']
    )
    
    logger.info(f"\nData shapes:")
    logger.info(f"  Stage 1: {stage1_data.shape}")
    logger.info(f"  Stage 2: {stage2_data.shape}")
    logger.info(f"  Stage 3: {stage3_data.shape}")
    logger.info(f"  Outcomes: {outcomes.shape}")
    logger.info(f"\nClass distribution: {dict(zip(*np.unique(outcomes, return_counts=True)))}")
    
    # Split data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Splitting Data")
    logger.info("=" * 70)
    
    indices = np.arange(len(outcomes))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=outcomes
    )
    
    X1_train, X1_test = stage1_data[train_idx], stage1_data[test_idx]
    X2_train, X2_test = stage2_data[train_idx], stage2_data[test_idx]
    X3_train, X3_test = stage3_data[train_idx], stage3_data[test_idx]
    y_train, y_test = outcomes[train_idx], outcomes[test_idx]
    
    logger.info(f"Training set: {len(train_idx)} samples")
    logger.info(f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"Test set: {len(test_idx)} samples")
    logger.info(f"  Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Train model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Training DMD Predictor")
    logger.info("=" * 70)
    logger.info(f"Initializing model with:")
    logger.info(f"  - DMD rank: {config['dmd_rank']}")
    logger.info(f"  - Classifier: {config['classifier_type']}")
    
    model = DMDPredictor(
        dmd_rank=config['dmd_rank'],
        classifier_type=config['classifier_type'],
        random_state=config['random_state']
    )
    
    logger.info("\nFitting model to training data...")
    model.fit(X1_train, X2_train, X3_train, y_train)
    logger.info("Model training completed!")
    
    # Evaluate model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Model Evaluation")
    logger.info("=" * 70)
    
    # Training set evaluation
    logger.info("\n--- Training Set Evaluation ---")
    y_train_pred = model.predict(X1_train, X2_train, X3_train)
    y_train_proba = model.predict_proba(X1_train, X2_train, X3_train)[:, 1]
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, verbose=True)
    
    # Test set evaluation
    logger.info("\n--- Test Set Evaluation ---")
    y_test_pred = model.predict(X1_test, X2_test, X3_test)
    y_test_proba = model.predict_proba(X1_test, X2_test, X3_test)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, verbose=True)
    
    # Feature importance
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Feature Importance")
    logger.info("=" * 70)
    
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        top_n = 10
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        logger.info(f"\nTop {top_n} most important features:")
        for i, idx in enumerate(top_indices, 1):
            logger.info(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")
    else:
        logger.info("Feature importance not available for this classifier.")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("\nCompetition Metrics (Test Set):")
    logger.info(f"  🎯 Primary Metric - AUC-ROC:     {test_metrics['auc_roc']:.4f}")
    logger.info(f"  🔹 Tie-breaker 1 - Log Loss:    {test_metrics['log_loss']:.4f}")
    logger.info(f"  🔹 Tie-breaker 2 - F1 Score:    {test_metrics['f1_score']:.4f}")
    logger.info(f"\nOther Metrics:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Example completed successfully!")
    logger.info("=" * 70)
    
    return model, test_metrics


if __name__ == '__main__':
    main()

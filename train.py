"""
Training pipeline for DMD-based depression prediction.

This script provides a complete training pipeline including:
- Data loading and preprocessing
- Model training with cross-validation
- Model evaluation
- Model saving
"""

import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import json

from warnd.models.dmd_model import DMDPredictor
from warnd.preprocessing.data_handler import DataHandler
from warnd.evaluation.metrics import evaluate_model, cross_validate_metrics
from warnd.utils.config import Config
from warnd.utils.logger import setup_logger


def train_model(config: Config, logger=None):
    """
    Train DMD-based depression prediction model.
    
    Args:
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Trained model and evaluation metrics
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info("Starting training pipeline...")
    
    # Initialize data handler
    data_handler = DataHandler(data_dir=config['data_dir'])
    
    # Load or create synthetic data
    logger.info("Loading data...")
    stage1_path = Path(config['data_dir']) / 'stage1.csv'
    stage2_path = Path(config['data_dir']) / 'stage2.csv'
    stage3_path = Path(config['data_dir']) / 'stage3.csv'
    outcomes_path = Path(config['data_dir']) / 'outcomes.csv'
    
    if all(p.exists() for p in [stage1_path, stage2_path, stage3_path, outcomes_path]):
        # Load real data
        data_handler.load_data(
            stage1_path=stage1_path,
            stage2_path=stage2_path,
            stage3_path=stage3_path,
            outcomes_path=outcomes_path
        )
        stage1_data, stage2_data, stage3_data, outcomes = data_handler.prepare_data()
    else:
        # Create synthetic data for demonstration
        logger.info("No data files found. Creating synthetic data for demonstration...")
        stage1_data, stage2_data, stage3_data, outcomes = data_handler.create_synthetic_data(
            n_participants=config.get('n_participants', 100),
            n_stage2_timesteps=config['stage2_timesteps'],
            n_stage1_features=config['stage1_features'],
            n_stage2_features=config['stage2_features'],
            n_stage3_features=config['stage3_features'],
            random_state=config['random_state']
        )
    
    logger.info(f"Data shapes: Stage1={stage1_data.shape}, Stage2={stage2_data.shape}, "
                f"Stage3={stage3_data.shape}, Outcomes={outcomes.shape}")
    
    # Split data
    logger.info("Splitting data into train/test sets...")
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
    
    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # Initialize and train model
    logger.info("Training DMD predictor...")
    model = DMDPredictor(
        dmd_rank=config['dmd_rank'],
        classifier_type=config['classifier_type'],
        random_state=config['random_state']
    )
    
    model.fit(X1_train, X2_train, X3_train, y_train)
    
    # Evaluate on training set
    logger.info("Evaluating on training set...")
    y_train_pred = model.predict(X1_train, X2_train, X3_train)
    y_train_proba = model.predict_proba(X1_train, X2_train, X3_train)[:, 1]
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, verbose=False)
    
    logger.info(f"Training - AUC-ROC: {train_metrics['auc_roc']:.4f}, "
                f"Log Loss: {train_metrics['log_loss']:.4f}, "
                f"F1: {train_metrics['f1_score']:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_test_pred = model.predict(X1_test, X2_test, X3_test)
    y_test_proba = model.predict_proba(X1_test, X2_test, X3_test)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, verbose=True)
    
    # Cross-validation
    if config.get('run_cv', False):
        logger.info(f"Running {config['n_folds']}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, 
                            random_state=config['random_state'])
        
        cv_y_true = []
        cv_y_pred = []
        cv_y_proba = []
        
        for fold, (train_cv_idx, val_cv_idx) in enumerate(cv.split(train_idx, y_train)):
            logger.info(f"  Fold {fold + 1}/{config['n_folds']}...")
            
            # Get fold data
            fold_train_idx = train_idx[train_cv_idx]
            fold_val_idx = train_idx[val_cv_idx]
            
            X1_fold_train = stage1_data[fold_train_idx]
            X2_fold_train = stage2_data[fold_train_idx]
            X3_fold_train = stage3_data[fold_train_idx]
            y_fold_train = outcomes[fold_train_idx]
            
            X1_fold_val = stage1_data[fold_val_idx]
            X2_fold_val = stage2_data[fold_val_idx]
            X3_fold_val = stage3_data[fold_val_idx]
            y_fold_val = outcomes[fold_val_idx]
            
            # Train fold model
            fold_model = DMDPredictor(
                dmd_rank=config['dmd_rank'],
                classifier_type=config['classifier_type'],
                random_state=config['random_state']
            )
            fold_model.fit(X1_fold_train, X2_fold_train, X3_fold_train, y_fold_train)
            
            # Predict on validation fold
            y_fold_pred = fold_model.predict(X1_fold_val, X2_fold_val, X3_fold_val)
            y_fold_proba = fold_model.predict_proba(X1_fold_val, X2_fold_val, X3_fold_val)[:, 1]
            
            cv_y_true.append(y_fold_val)
            cv_y_pred.append(y_fold_pred)
            cv_y_proba.append(y_fold_proba)
        
        # Compute CV metrics
        cv_metrics = cross_validate_metrics(cv_y_true, cv_y_pred, cv_y_proba)
        
        logger.info("\nCross-validation results:")
        logger.info(f"  AUC-ROC: {cv_metrics['auc_roc'][0]:.4f} ± {cv_metrics['auc_roc'][1]:.4f}")
        logger.info(f"  Log Loss: {cv_metrics['log_loss'][0]:.4f} ± {cv_metrics['log_loss'][1]:.4f}")
        logger.info(f"  F1 Score: {cv_metrics['f1_score'][0]:.4f} ± {cv_metrics['f1_score'][1]:.4f}")
    
    # Save model
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'dmd_model.pkl'
    logger.info(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    logger.info(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, 'w') as f:
        json.dump({
            'train': {k: float(v) for k, v in train_metrics.items()},
            'test': {k: float(v) for k, v in test_metrics.items()}
        }, f, indent=2)
    
    # Save config
    config_path = output_dir / 'config.json'
    config.save(config_path)
    
    logger.info("Training pipeline completed successfully!")
    
    return model, test_metrics


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Train DMD-based depression prediction model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--dmd-rank', type=int, help='DMD rank for truncation')
    parser.add_argument('--classifier', type=str, default='gradient_boosting',
                       choices=['gradient_boosting', 'random_forest'],
                       help='Classifier type')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override with command-line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.dmd_rank:
        config['dmd_rank'] = args.dmd_rank
    if args.classifier:
        config['classifier_type'] = args.classifier
    if args.cv:
        config['run_cv'] = True
    if args.n_folds:
        config['n_folds'] = args.n_folds
    if args.random_state:
        config['random_state'] = args.random_state
    
    # Set up logger
    logger = setup_logger(log_file=str(Path(config['output_dir']) / 'training.log'))
    
    # Train model
    model, metrics = train_model(config, logger)
    
    return model, metrics


if __name__ == '__main__':
    main()

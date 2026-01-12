"""
Prediction script for DMD-based depression prediction model.

Load a trained model and make predictions on new data.
"""

import numpy as np
import argparse
import pickle
from pathlib import Path
import pandas as pd

from warnd.preprocessing.data_handler import DataHandler
from warnd.utils.logger import setup_logger


def predict(model_path, data_dir, output_path, logger=None):
    """
    Make predictions using trained model.
    
    Args:
        model_path: Path to trained model pickle file
        data_dir: Directory containing data files
        output_path: Path to save predictions
        logger: Logger instance
        
    Returns:
        Predictions array
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info("Loading trained model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info("Loading data...")
    data_handler = DataHandler(data_dir=data_dir)
    
    stage1_path = Path(data_dir) / 'stage1.csv'
    stage2_path = Path(data_dir) / 'stage2.csv'
    stage3_path = Path(data_dir) / 'stage3.csv'
    
    if all(p.exists() for p in [stage1_path, stage2_path, stage3_path]):
        data_handler.load_data(
            stage1_path=stage1_path,
            stage2_path=stage2_path,
            stage3_path=stage3_path
        )
        stage1_data, stage2_data, stage3_data, _ = data_handler.prepare_data()
    else:
        raise ValueError(f"Data files not found in {data_dir}")
    
    logger.info("Making predictions...")
    predictions = model.predict(stage1_data, stage2_data, stage3_data)
    probabilities = model.predict_proba(stage1_data, stage2_data, stage3_data)[:, 1]
    
    # Save predictions
    logger.info(f"Saving predictions to {output_path}...")
    output_df = pd.DataFrame({
        'participant_id': data_handler.participant_ids,
        'prediction': predictions,
        'probability': probabilities
    })
    output_df.to_csv(output_path, index=False)
    
    logger.info("Prediction completed successfully!")
    logger.info(f"Predicted {np.sum(predictions == 1)} positive cases out of {len(predictions)} total")
    
    return predictions, probabilities


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Make predictions using trained DMD model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='Output predictions file')
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    
    # Make predictions
    predictions, probabilities = predict(args.model, args.data_dir, args.output, logger)
    
    return predictions, probabilities


if __name__ == '__main__':
    main()

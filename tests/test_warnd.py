"""
Unit tests for WARND DMD-based depression prediction model.
"""

import unittest
import numpy as np
from warnd.models.dmd_core import DMDCore
from warnd.models.dmd_model import DMDPredictor
from warnd.preprocessing.data_handler import DataHandler
from warnd.preprocessing.feature_extractor import FeatureExtractor
from warnd.evaluation.metrics import evaluate_model, compute_metrics, compare_models
from warnd.utils.config import Config


class TestDMDCore(unittest.TestCase):
    """Test DMD core functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create simple time series data
        t = np.linspace(0, 10, 50)
        n_features = 5
        self.X = np.array([np.sin(i * t) + 0.1 * np.random.randn(len(t)) 
                          for i in range(n_features)])
    
    def test_dmd_initialization(self):
        """Test DMD initialization."""
        dmd = DMDCore(rank=3)
        self.assertEqual(dmd.rank, 3)
        self.assertIsNone(dmd.modes)
    
    def test_dmd_fit(self):
        """Test DMD fitting."""
        dmd = DMDCore(rank=3)
        dmd.fit(self.X)
        
        self.assertIsNotNone(dmd.modes)
        self.assertIsNotNone(dmd.eigenvalues)
        self.assertIsNotNone(dmd.amplitudes)
        self.assertIsNotNone(dmd.omega)
    
    def test_dmd_reconstruct(self):
        """Test DMD reconstruction."""
        dmd = DMDCore(rank=3)
        dmd.fit(self.X)
        
        reconstruction = dmd.reconstruct(n_timesteps=self.X.shape[1])
        
        self.assertEqual(reconstruction.shape, self.X.shape)
    
    def test_dmd_predict(self):
        """Test DMD prediction."""
        dmd = DMDCore(rank=3)
        dmd.fit(self.X)
        
        prediction = dmd.predict(n_steps=10)
        
        self.assertEqual(prediction.shape[0], self.X.shape[0])
        self.assertEqual(prediction.shape[1], 10)
    
    def test_dmd_extract_features(self):
        """Test DMD feature extraction."""
        dmd = DMDCore(rank=3)
        features = dmd.extract_features(self.X)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)


class TestDMDPredictor(unittest.TestCase):
    """Test DMD predictor."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data_handler = DataHandler()
        self.stage1, self.stage2, self.stage3, self.outcomes = \
            self.data_handler.create_synthetic_data(n_participants=50)
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = DMDPredictor(dmd_rank=5, classifier_type='gradient_boosting')
        self.assertEqual(predictor.dmd_rank, 5)
        self.assertFalse(predictor.is_fitted)
    
    def test_predictor_fit(self):
        """Test predictor fitting."""
        predictor = DMDPredictor(dmd_rank=5)
        predictor.fit(self.stage1, self.stage2, self.stage3, self.outcomes)
        
        self.assertTrue(predictor.is_fitted)
    
    def test_predictor_predict(self):
        """Test predictor predictions."""
        predictor = DMDPredictor(dmd_rank=5)
        predictor.fit(self.stage1, self.stage2, self.stage3, self.outcomes)
        
        predictions = predictor.predict(self.stage1, self.stage2, self.stage3)
        
        self.assertEqual(len(predictions), len(self.outcomes))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_predictor_predict_proba(self):
        """Test predictor probability predictions."""
        predictor = DMDPredictor(dmd_rank=5)
        predictor.fit(self.stage1, self.stage2, self.stage3, self.outcomes)
        
        probabilities = predictor.predict_proba(self.stage1, self.stage2, self.stage3)
        
        self.assertEqual(probabilities.shape, (len(self.outcomes), 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))


class TestDataHandler(unittest.TestCase):
    """Test data handler."""
    
    def test_synthetic_data_creation(self):
        """Test synthetic data creation."""
        handler = DataHandler()
        stage1, stage2, stage3, outcomes = handler.create_synthetic_data(
            n_participants=100,
            n_stage2_timesteps=354,
            n_stage1_features=20,
            n_stage2_features=30,
            n_stage3_features=15
        )
        
        self.assertEqual(stage1.shape, (100, 20))
        self.assertEqual(stage2.shape, (100, 354, 30))
        self.assertEqual(stage3.shape, (100, 8, 15))
        self.assertEqual(outcomes.shape, (100,))
        self.assertTrue(all(o in [0, 1] for o in outcomes))
    
    def test_synthetic_data_class_balance(self):
        """Test that synthetic data has both classes."""
        handler = DataHandler()
        _, _, _, outcomes = handler.create_synthetic_data(n_participants=100)
        
        unique_classes = np.unique(outcomes)
        self.assertEqual(len(unique_classes), 2)
        self.assertIn(0, unique_classes)
        self.assertIn(1, unique_classes)


class TestFeatureExtractor(unittest.TestCase):
    """Test feature extractor."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.extractor = FeatureExtractor()
        self.time_series = np.random.randn(100, 10)
    
    def test_temporal_features(self):
        """Test temporal feature extraction."""
        features = self.extractor.extract_temporal_features(self.time_series)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_frequency_features(self):
        """Test frequency feature extraction."""
        features = self.extractor.extract_frequency_features(self.time_series)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_complexity_features(self):
        """Test complexity feature extraction."""
        features = self.extractor.extract_complexity_features(self.time_series)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_stage2_features(self):
        """Test Stage 2 feature extraction."""
        features = self.extractor.extract_stage2_features(
            self.time_series,
            include_temporal=True,
            include_frequency=True,
            include_complexity=True
        )
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def setUp(self):
        """Set up test predictions."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        self.y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.7, 0.2, 0.85, 0.6])
    
    def test_compute_metrics(self):
        """Test metric computation."""
        metrics = compute_metrics(self.y_true, self.y_pred, self.y_proba)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('auc_roc', metrics)
        self.assertIn('log_loss', metrics)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        metrics = evaluate_model(self.y_true, self.y_pred, self.y_proba, verbose=False)
        
        self.assertIsInstance(metrics, dict)
        self.assertGreater(metrics['auc_roc'], 0)
        self.assertLessEqual(metrics['auc_roc'], 1)
    
    def test_compare_models(self):
        """Test model comparison."""
        metrics1 = {'auc_roc': 0.8, 'log_loss': 0.5, 'f1_score': 0.75}
        metrics2 = {'auc_roc': 0.7, 'log_loss': 0.6, 'f1_score': 0.70}
        
        result = compare_models(metrics1, metrics2, "Model 1", "Model 2")
        
        self.assertIn("Model 1", result)
        self.assertIn("AUC-ROC", result)


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test config initialization."""
        config = Config()
        
        # dmd_rank can be None (means full rank)
        self.assertIn('dmd_rank', config.config)
        self.assertIsNotNone(config['classifier_type'])
    
    def test_config_update(self):
        """Test config update."""
        config = Config()
        config['dmd_rank'] = 10
        
        self.assertEqual(config['dmd_rank'], 10)
    
    def test_config_to_dict(self):
        """Test config to dictionary."""
        config = Config({'test_param': 123})
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['test_param'], 123)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test full training and prediction pipeline."""
        np.random.seed(42)
        
        # Create data
        data_handler = DataHandler()
        stage1, stage2, stage3, outcomes = data_handler.create_synthetic_data(
            n_participants=100
        )
        
        # Split data
        n_train = 80
        X1_train, X1_test = stage1[:n_train], stage1[n_train:]
        X2_train, X2_test = stage2[:n_train], stage2[n_train:]
        X3_train, X3_test = stage3[:n_train], stage3[n_train:]
        y_train, y_test = outcomes[:n_train], outcomes[n_train:]
        
        # Train model
        model = DMDPredictor(dmd_rank=5, classifier_type='gradient_boosting')
        model.fit(X1_train, X2_train, X3_train, y_train)
        
        # Predict
        predictions = model.predict(X1_test, X2_test, X3_test)
        probabilities = model.predict_proba(X1_test, X2_test, X3_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, predictions, probabilities[:, 1], verbose=False)
        
        # Check metrics are in valid ranges
        self.assertGreaterEqual(metrics['auc_roc'], 0)
        self.assertLessEqual(metrics['auc_roc'], 1)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertGreaterEqual(metrics['f1_score'], 0)
        self.assertLessEqual(metrics['f1_score'], 1)


if __name__ == '__main__':
    unittest.main()

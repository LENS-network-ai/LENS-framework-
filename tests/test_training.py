#!/usr/bin/env python
# coding: utf-8

import unittest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from analysis import analyze_overfitting, calculate_class_weights

class TestAnalysisFunctions(unittest.TestCase):
    def test_analyze_overfitting(self):
        """Test the overfitting analysis function"""
        # Perfect generalization - no gap
        train_accs = [0.7, 0.8, 0.9]
        val_accs = [0.7, 0.8, 0.9]
        result = analyze_overfitting(train_accs, val_accs, warmup_epochs=0)
        self.assertEqual(result["severity"], "None")
        self.assertEqual(result["avg_gap"], 0.0)
        
        # Severe overfitting - big gap
        train_accs = [0.7, 0.9, 1.0]
        val_accs = [0.7, 0.7, 0.7]
        result = analyze_overfitting(train_accs, val_accs, warmup_epochs=0)
        self.assertEqual(result["severity"], "Severe")
        self.assertGreater(result["avg_gap"], 0.1)

    def test_class_weights(self):
        """Test class weight calculation"""
        # Create a mock dataset with imbalanced classes
        dataset = MagicMock()
        train_idx = list(range(10))
        
        # Mock dataset items with imbalanced labels [0, 0, 0, 0, 0, 0, 0, 1, 1, 2]
        dataset.__getitem__ = lambda self, idx: {'label': 0 if idx < 7 else (1 if idx < 9 else 2)}
        
        weights = calculate_class_weights(dataset, train_idx)
        
        # Check weights type and shape
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(len(weights), 3)  # 3 classes
        
        # Check that minority classes get higher weights
        self.assertLess(weights[0], weights[1])  # Class 0 (7 samples) < class 1 (2 samples)
        self.assertLess(weights[1], weights[2])  # Class 1 (2 samples) < class 2 (1 sample)

class TestTrainingHelpers(unittest.TestCase):
    def setUp(self):
        """Set up temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
    
    @patch('training_loop.train_epoch')
    @patch('training_loop.validate_epoch')
    def test_train_and_evaluate(self, mock_validate, mock_train):
        """Test the train_and_evaluate function with mocks"""
        from training_loop import train_and_evaluate
        
        # Set up mocks
        mock_train.side_effect = [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]  # (acc, loss) tuples
        mock_validate.side_effect = [(0.6, 0.4, 20.0), (0.7, 0.3, 25.0), (0.75, 0.25, 30.0)]  # (acc, loss, sparsity)
        
        # Create mock model and data
        model = MagicMock()
        model.num_classes = 3
        model.edge_stats = [0.5, 0.4, 0.3]  # Mock edge statistics
        
        # Execute with minimum required arguments
        results = train_and_evaluate(
            model=model,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            device=torch.device("cpu"),
            num_epochs=3,
            n_features=512,
            output_dir=self.test_dir
        )
        
        # Check results structure
        self.assertIn("train_accs", results)
        self.assertIn("val_accs", results)
        self.assertIn("best_val_acc", results)
        self.assertIn("best_epoch", results)
        
        # Check that best model is correctly identified
        self.assertEqual(results["best_val_acc"], 0.75)
        self.assertEqual(results["best_epoch"], 3)
        self.assertEqual(results["best_edge_sparsity"], 30.0)
        
        # Check that train and validate were called the correct number of times
        self.assertEqual(mock_train.call_count, 3)
        self.assertEqual(mock_validate.call_count, 3)

if __name__ == '__main__':
    unittest.main()

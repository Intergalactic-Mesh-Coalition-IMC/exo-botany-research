import unittest
from model_evaluation.model_evaluation import evaluate_model

class TestModelEvaluation(unittest.TestCase):
    def test_evaluate_model(self):
        # Test data
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]

        # Expected output
        expected_output = {'accuracy': 0.5, 'f1_score': 0.5, 'roc_auc': 0.5}

        # Test evaluate_model function
        result = evaluate_model(y_true, y_pred)
        self.assertDictEqual(result, expected_output)

    def test_evaluate_model_multiclass(self):
        # Test data
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]

        # Expected output
        expected_output = {'accuracy': 1.0, 'f1_score': 1.0, 'roc_auc': 1.0}

        # Test evaluate_model function with multiclass classification
        result = evaluate_model(y_true, y_pred)
        self.assertDictEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()

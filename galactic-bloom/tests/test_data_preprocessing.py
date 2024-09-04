import unittest
from unittest.mock import patch
from preprocessing.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        # Test data
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

        # Expected output
        expected_data = pd.DataFrame({'feature1': [0.5, 1.0, 1.5], 'feature2': [2.0, 2.5, 3.0]})

        # Test preprocess_data function
        result = preprocess_data(data)
        pd.testing.assert_frame_equal(result, expected_data)

    @patch('preprocessing.data_preprocessing.StandardScaler')
    def test_preprocess_data_scaler(self, mock_scaler):
        # Test data
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

        # Mock scaler
        mock_scaler.return_value.fit_transform.return_value = pd.DataFrame({'feature1': [0.5, 1.0, 1.5], 'feature2': [2.0, 2.5, 3.0]})

        # Test preprocess_data function
        result = preprocess_data(data)
        pd.testing.assert_frame_equal(result, pd.DataFrame({'feature1': [0.5, 1.0, 1.5], 'feature2': [2.0, 2.5, 3.0]}))

if __name__ == '__main__':
    unittest.main()

import unittest
from feature_engineering.feature_engineering import engineer_features

class TestFeatureEngineering(unittest.TestCase):
    def test_engineer_features(self):
        # Test data
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

        # Expected output
        expected_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'feature3': [5, 7, 9]})

        # Test engineer_features function
        result = engineer_features(data)
        pd.testing.assert_frame_equal(result, expected_data)

    def test_engineer_features_missing_values(self):
        # Test data
        data = pd.DataFrame({'feature1': [1, 2, np.nan], 'feature2': [4, 5, 6]})

        # Expected output
        expected_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'feature3': [5, 7, 9]})

        # Test engineer_features function with missing values
        result = engineer_features(data)
        pd.testing.assert_frame_equal(result, expected_data)

if __name__ == '__main__':
    unittest.main()

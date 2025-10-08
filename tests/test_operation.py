import unittest
import numpy as np
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation

class TestMlProject(unittest.TestCase):
    def test_data_ingestion(self):
        obj = DataIngestion()
        val1, val2 = obj.initiate_data_ingestion()
        
        # Check both paths
        self.assertTrue(val1.endswith('.csv'), f"{val1} does not end with .csv")
        self.assertTrue(val2.endswith('.csv'), f"{val2} does not end with .csv")

    def test_data_transformation(self):
        obj = DataTransformation()
        val1, val2, val3 = obj.initiate_data_transformation(
            'artifacts/train.csv', 'artifacts/test.csv')
        self.assertEqual(val1.shape[0], 800)
        self.assertEqual(val2.shape[0], 200)
        self.assertEqual(val1.shape[1], val2.shape[1])
        self.assertTrue(val3.endswith('.pkl'), 
                        f"{val1} does not end with .pkl")
        self.assertTrue(type(val1) is np.ndarray, 
                        f"{type(val1)} is not numpy ndarray")
        self.assertEqual(val2.ndim, 2)

if __name__ == "__main__":
    unittest.main()
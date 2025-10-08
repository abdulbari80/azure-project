import os
import unittest
import numpy as np
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.utils import load_object

class TestMlProject(unittest.TestCase):
    def test_data_ingestion(self):
        obj = DataIngestion()
        val1, val2 = obj.initiate_data_ingestion()
        
        # Check both paths
        self.assertTrue(val1.endswith('.csv'), f"{val1} does not end with .csv")
        self.assertTrue(val2.endswith('.csv'), f"{val2} does not end with .csv")

    def test_data_transformation(self):
        obj = DataTransformation()
        train_path = os.path.join('artifacts','train.csv')
        test_path = os.path.join('artifacts','test.csv')
        val1, val2, val3 = obj.initiate_data_transformation(
            train_path, test_path)
        self.assertEqual(val1.shape[0], 800)
        self.assertEqual(val2.shape[0], 200)
        self.assertEqual(val1.shape[1], val2.shape[1])
        self.assertTrue(val3.endswith('.pkl'), 
                        f"{val1} does not end with .pkl")
        self.assertTrue(type(val1) is np.ndarray, 
                        f"{type(val1)} is not numpy ndarray")
        self.assertEqual(val2.ndim, 2)

    def test_load_object(self):
        model_path = os.path.join('artifacts', 'model_v1.0.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        obj = load_object(preprocessor_path)
        self.assertIsNotNone(obj, "Loaded object is None")
        self.assertTrue(hasattr(obj, 'transform'), 
                        "Loaded object does not have 'transform' method")
        self.assertIsNotNone(load_object(model_path), 
                          "Loaded model object is None")

if __name__ == "__main__":
    unittest.main()
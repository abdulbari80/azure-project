import os
import unittest
import numpy as np
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.utils import load_object

class TestMlProject(unittest.TestCase):
    def test_data_transformation(self):
        obj = DataTransformation()
        train_path = os.path.join('artifacts','train.csv')
        test_path = os.path.join('artifacts','test.csv')
        val1, val2, val3 = obj.initiate_data_transformation(
            train_path, test_path)
        self.assertIsNotNone(val1, "Transformed train array is None")
        self.assertIsNotNone(val2, "Transformed test array is None")
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
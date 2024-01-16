import unittest
import numpy as np
from data_preparation import DataPreparation

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.data_scaled = np.random.rand(3382, 1) 
        self.data_prep = DataPreparation(self.data_scaled)

    def test_prepare_data(self):
        x_train, y_train, x_test, y_test = self.data_prep.prepare_data()

        self.assertEqual(x_train.shape, (2830, 70))
        self.assertEqual(y_train.shape, (2830,))
        self.assertEqual(x_test.shape, (481, 70))
        self.assertEqual(y_test.shape, (481,))

if __name__ == '__main__':
    unittest.main()
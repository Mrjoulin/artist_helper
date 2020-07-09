import numpy as np
import unittest
import cv2
import os

# Local import
from run import Model


image_shape = tuple([int(i) for i in os.getenv('IMAGES_SHAPE').split(',')])


class MyTestCase(unittest.TestCase):
    def test_prediction(self):
        print("Run test prediction")
        test_image_path = 'images/test.jpg'
        test_image_label = 'акрилом'
        # Load image from path
        test_image = cv2.imread(test_image_path)
        # Load model
        model = Model()

        predict = model.prediction(test_image)

        self.assertEqual(predict['names'][0], test_image_label)


if __name__ == '__main__':
    print("Run tests")
    unittest.main()

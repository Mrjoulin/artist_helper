from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import tensorflow as tf

from skimage.transform import resize
from skimage.io import imread
import numpy as np
import argparse
import logging
import json
import time
import os

image_shape = tuple([int(i) for i in os.getenv('IMAGES_SHAPE').split(',')])
classes_file = str(os.getenv('IMAGES_CLASSES_FILE'))
images_types = os.getenv("IMAGES_TYPES").split(',')

classes = []
with open(classes_file, 'r') as f:
    classes_info = json.load(f)
    for images_type in images_types:
        if images_type in classes_info:
            classes.append(classes_info[images_type])


class Model:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(os.getenv('MODEL_DIR')) + str(os.getenv("MODEL_NAME"))

        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)

        try:
            self.model = load_model(model_path)
        except Exception as e:
            logging.info('Cannot load model ' + str(args.model) + '. Exception: ' + str(e))
            exit(2)

    def prediction(self, test_image):
        with self.graph.as_default():
            set_session(self.sess)
            # prepare image
            test_image = resize(test_image, image_shape[:2])
            test_image = test_image[:, :, ::-1]
            test_image = np.array([test_image])
            # get prediction

            start = time.time()
            predict = self.model.predict(test_image)[0]
            prediction_class = int(np.argmax(predict))

            logging.info(
                'Prediction time {pr_time}. Prediction result: {pr_res}. All prediction: {all}'
                .format(
                    pr_time=time.time() - start,
                    pr_res=classes[0][prediction_class],
                    all=predict
                )
            )

            predict = list(predict * 100)
            sorted_predict = sorted(predict, reverse=True)

            return {
                "prediction": sorted_predict,
                "names": [
                    classes[0][predict.index(predict_class)] for predict_class in sorted_predict
                ]
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, default='./test.jpg', help='Path an image to render')
    parser.add_argument('--model', '-m', default='./model.h5', help='Path to you model .h5')
    args = parser.parse_args()

    image = imread(args.image)

    model = Model(model_path=args.model)

    model.prediction(image)

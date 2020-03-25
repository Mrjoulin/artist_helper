from tensorflow.python.keras.models import load_model

from cv2.cv2 import imread, resize
import numpy as np
import argparse
import logging
import time
import json
import os

from backend.train import printProgressBar

logging.basicConfig(
    format='[%(asctime)s: %(filename)s:%(lineno)s - %(funcName)10s()]%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
CONFIG_FILE = './config.json'

with open(CONFIG_FILE, 'r') as config_file:
    args = json.load(config_file)

path_to_images = args['images_path']
classes = os.listdir(path_to_images)
logging.info('Classes materials: ' + str(classes))
images_shape = tuple(args['images_shape'])
images_restrictions = args['images_restrictions']
percent_to_test = args['percent_to_test']
sort_images = bool(args['sort_images'])


def test_dataset():
    test_images = []
    test_labels = []
    for image_class in classes:
        images_paths = os.listdir(os.path.join(path_to_images, image_class))
        if sort_images:
            logging.info('Sort images')
            images_paths.sort()
        if len(images_paths) > images_restrictions:
            images_paths = images_paths[int(images_restrictions * (1 - percent_to_test)):]
        else:
            images_paths = images_paths[int(len(images_paths) * (1 - percent_to_test)):]
        errors = 0

        logging.info('Start prepare %s. Done: %s/%s' % (image_class, classes.index(image_class), len(classes)))

        printProgressBar(0, len(images_paths), prefix='Test images progress:', suffix='Complete')
        for num_image in range(len(images_paths)):
            try:
                image = imread(os.path.join(path_to_images, image_class, images_paths[num_image]))
                image = resize(image, images_shape[:2])
                test_images.append(image)
                printProgressBar(num_image + 1, len(images_paths), prefix='Test images progress:', suffix='Complete')
            except Exception as ex:
                logging.info("GET EXCEPTION: " + str(ex))
                errors += 1
        test_labels += [classes.index(image_class)] * (len(images_paths) - errors)

        logging.info("Train images: " + str(len(test_images)) + ", train labels: " + str(len(test_labels)))

    assert len(test_labels) == len(test_images)

    num_test_images = len(test_images)
    logging.info("Generate test dataset with %s images" % num_test_images)

    perm = np.random.permutation(num_test_images)

    test_images = np.array([test_images])[0][perm]
    test_images = test_images / 255.0
    test_labels = np.array([test_labels])[0][perm]

    logging.info("Test images dataset shape: " + str(test_images.shape))
    logging.info("Test labels dataset shape: " + str(test_labels.shape))

    return test_images, test_labels


def test():
    logging.info('Generate dataset')
    test_images, test_labels = test_dataset()
    predictions = []

    for image, label in zip(test_images, test_labels):
        image = np.array([image])
        start = time.time()
        predict = model.predict(image)
        prediction_class = int(np.argmax(predict))
        logging.info(
            'Prediction time {pr_time}. Prediction result: {pr_res}. Correct result: {cor_res}. All prediction: {all}'
            .format(
                pr_time=time.time() - start,
                pr_res=classes[prediction_class],
                cor_res=classes[label],
                all=predict
            )
        )
        predictions.append(prediction_class == label)
    logging.info('Accuracy: %s' % (sum(predictions) / len(predictions)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='./models/model.h5', help='Path to you model .h5')
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except Exception as e:
        logging.info('Cannot load model ' + str(args.model) + '. Exception: ' + str(e))
        exit(2)

    test()

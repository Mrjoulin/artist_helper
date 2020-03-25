from tensorflow.python.keras.layers import *
from tensorflow import keras
import tensorflow as tf

from cv2.cv2 import imread, resize
import numpy as np
import argparse
import datetime
import logging
import json
import os

logging.basicConfig(
    format='[%(asctime)s: %(filename)s:%(lineno)s - %(funcName)10s()]%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# classes = ["watercolor", "pencil", "coal", "sangina|sepia", "oil", "gouache", "pen", "markers", "acrylic", "tempera"]
MODEL_NAME = './backend/models/new_model' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'
CONFIG_FILE = './config.json'

with open(CONFIG_FILE, 'r') as config_file:
    args = json.load(config_file)

path_to_images = args['images_path']
classes = os.listdir(path_to_images)
logging.info('Classes materials: ' + str(classes))
images_shape = tuple(args['images_shape'])
images_restrictions = args['images_restrictions']
percent_to_test = args['percent_to_test']
sort_images = args['sort_images']


# Print iterations progress
def printProgressBar (iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def prepare_data():
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for image_class in classes:
        images_paths = os.listdir(os.path.join(path_to_images, image_class))
        if sort_images:
            logging.info('Sort images')
            images_paths.sort()
        # Limit on the number of images
        images_paths = images_paths[:images_restrictions]
        train_images_errors = test_images_errors = 0
        num_images_to_train = int(len(images_paths) * (1 - percent_to_test))
        logging.info('Start prepare %s. Done: %s/%s' % (image_class, classes.index(image_class), len(classes)))

        printProgressBar(0, num_images_to_train, prefix='Train images progress:', suffix='Complete')
        for num_image in range(num_images_to_train):
            try:
                image = imread(os.path.join(path_to_images, image_class, images_paths[num_image]))
                image = resize(image, images_shape[:2])
                train_images.append(image)
                printProgressBar(num_image + 1, num_images_to_train, prefix='Train images progress:', suffix='Complete')
            except Exception as e:
                logging.info("GET EXCEPTION: " + str(e))
                train_images_errors += 1
        train_labels += [classes.index(image_class)] * (num_images_to_train - train_images_errors)
        logging.info("Train images: " + str(len(train_images)) + ", train labels: " + str(len(train_labels)))

        printProgressBar(0, num_images_to_train, prefix='Test images progress:', suffix='Complete')
        for num_image in range(len(images_paths) - num_images_to_train):
            try:
                image = imread(os.path.join(path_to_images, image_class, images_paths[-1 - num_image]))
                image = resize(image, images_shape[:2])
                test_images.append(image)
                printProgressBar(num_image + 1, len(images_paths) - num_images_to_train,
                                 prefix='Test images progress:', suffix='Complete')
            except Exception as e:
                logging.info("GET EXCEPTION: " + str(e))
                test_images_errors += 1
        test_labels += [classes.index(image_class)] * (len(images_paths) - num_images_to_train - test_images_errors)
        logging.info("Test images: " + str(len(test_images)) + ", test labels: " + str(len(test_labels)))

    assert len(train_labels) == len(train_images)
    assert len(test_labels) == len(test_images)

    num_train_images = len(train_images)
    logging.info("Generate train dataset with %s images" % num_train_images)

    perm = np.random.permutation(num_train_images)

    train_images = np.array([train_images])[0][perm]
    train_images = train_images / 255.0
    train_labels = np.array([train_labels])[0][perm]

    num_test_images = len(test_images)
    logging.info("Generate test dataset with %s images" % num_test_images)

    perm = np.random.permutation(num_test_images)

    test_images = np.array([test_images])[0][perm]
    test_images = test_images / 255.0
    test_labels = np.array([test_labels])[0][perm]

    logging.info("Train images dataset shape: " + str(train_images.shape))
    logging.info("Train labels dataset shape: " + str(train_labels.shape))
    logging.info("Test images dataset shape: " + str(test_images.shape))
    logging.info("Test labels dataset shape: " + str(test_labels.shape))

    return train_images, train_labels, test_images, test_labels


def init_model(shape):

    model = keras.Sequential([
        Conv2D(16, (3, 3), padding="same", input_shape=shape, activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.25),

        Conv2D(32, (3, 3), padding="same", activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(32, (3, 3), padding="same", activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        BatchNormalization(),

        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(epochs=20, model_path=None):
    # Add tensorboard output
    log_dir = "./backend/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    train_images, train_labels, test_images, test_labels = prepare_data()

    if model_path is None:
        model = init_model(images_shape)
    else:
        model = keras.models.load_model(model_path)

    logging.info("Start training")
    model.fit(train_images, train_labels, epochs=epochs, callbacks=[tensorboard_callback])

    logging.info('Save model to ' + MODEL_NAME)
    model.save(MODEL_NAME)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    logging.info("Test Loss: %s" % test_loss)
    logging.info("Test accuracy: %s" % test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=28, help='Number of epochs in training')
    parser.add_argument('--model', '-m', default=None, help='Path to you model .h5')
    args = parser.parse_args()

    logging.info('Train artist helper model v.0.0.1')
    train(epochs=args.epochs, model_path=args.model)

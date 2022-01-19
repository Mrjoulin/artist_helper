from tensorflow.python.compiler.mlcompute import mlcompute
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.regularizers import *
from tensorflow import keras
import tensorflow as tf

import threading as thr
from PIL import Image
import subprocess
import argparse
import datetime
import logging
import json
import os

# Local import
from utils import *

logging.basicConfig(
    format='[%(asctime)s: %(filename)s:%(lineno)s - %(funcName)10s()]%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


# Re-writen to tensorflow v2.x.x

tf.compat.v1.disable_eager_execution()
mlcompute.set_mlc_device(device_name='gpu')

logging.info("is_apple_mlc_enabled %s" % mlcompute.is_apple_mlc_enabled())
logging.info("is_tf_compiled_with_apple_mlc %s" % mlcompute.is_tf_compiled_with_apple_mlc())
logging.info(f"eagerly? {tf.executing_eagerly()}")
logging.info(tf.config.list_logical_devices())


# List of tuples with models names and trained models history:
# [
#   (<model name>, <model history>),
#   ...
# ]
MODELS = []

# Set constants
try:
    images_shape = tuple([int(i) for i in os.getenv('IMAGES_SHAPE').split(',')])
    images_restrictions = int(os.getenv("IMAGES_RESTRICTIONS"))
    percent_to_test = float(os.getenv("PERCENT_TO_TEST"))
    shuffle_images = bool(os.getenv("SUFFLE_IMAGES"))
except Exception:
    images_shape = (200, 200, 3)
    images_restrictions = 700
    percent_to_test = 0.1
    shuffle_images = True

l2_regularizer_coef = 1e-4


# Check all data installed
def check_data(path_to_images: str):
    # Add images duplication
    images_duplication = {}
    nums_images = []
    for class_name in classes:
        _images = sorted(os.listdir(os.path.join(path_to_images, class_name)))
        all_images = []
        nums_images.append(len(_images))
        for img_name in _images:
            img_path = os.path.join(path_to_images, class_name, img_name)
            try:
                new_img = Image.open(img_path)
                if new_img not in all_images:
                    all_images.append(new_img)
                else:
                    print('Remove duble image:', img_path)
                    os.remove(img_path)
            except Exception as e:
                print('Remove:', img_path, '(Exception: %s)' % str(e))
                input()
                os.remove(img_path)

        images_duplication[class_name] = round(600 / len(_images))
        print(class_name, '-', len(_images))

    print('Mean:', np.mean(nums_images))
    print('Sum:', np.sum(nums_images))
    print("Images Dublication:", images_duplication)

    return images_duplication


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
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


# Shuffle all data in images and labels
def shuffle_dataset(images, labels):
    num_images = len(images)
    print("Generate dataset with %s images" % num_images)

    perm = np.random.permutation(num_images)

    images = np.array([images])[0][perm]
    images = images / 255.0
    labels = np.array([labels])[0][perm]

    return images, labels


def check_gpu_work():
    subprocess.call(['nvidia-smi'])


def prepare_data(path_to_images: str):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for image_class in classes:
        images_paths = os.listdir(os.path.join(path_to_images, image_class))
        # Limit on the number of images
        images_paths = images_paths[:images_restrictions]
        # List with all class images
        images = []
        print('Start prepare %s. Done: %s/%s' % (image_class, classes.index(image_class), len(classes)))

        for num_image, img_name in enumerate(images_paths):
            try:
                image = cv2.imread(os.path.join(path_to_images, image_class, img_name))
                image = cv2.resize(image, images_shape[:2])[:, :, ::-1]

                images.append(image)

                if images_duplication[image_class] > 1:
                    # Horizontal flip
                    images.append(cv2.flip(image, 1))

                if images_duplication[image_class] > 2:
                    # Converting image to LAB Color model
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    # Splitting the LAB image to different channels
                    l, a, b = cv2.split(lab)
                    # Applying CLAHE to L-channel
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
                    # Merge the CLAHE enhanced L-channel with the a and b channel
                    limg = cv2.merge((clahe, a, b))
                    # Converting image from LAB Color model to RGB model
                    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

                    images.append(image)

                printProgressBar(num_image + 1, len(images_paths), prefix='Load images progress:', suffix='Complete')
            except Exception as e:
                print("GET EXCEPTION: " + str(e))

        # Generate lables with the same lenght as images
        labels = [classes.index(image_class)] * len(images)
        print("Load images: " + str(len(images)) + ", load labels: " + str(len(labels)))

        if shuffle_images:
            print('Suffle images')
            np.random.shuffle(images)

        num_to_train = int(len(images) * (1 - percent_to_test))
        train_images += images[:num_to_train]
        train_labels += labels[:num_to_train]
        test_images += images[num_to_train:]
        test_labels += labels[num_to_train:]

    assert len(train_labels) == len(train_images)
    assert len(test_labels) == len(test_images)

    train_images, train_labels = shuffle_dataset(train_images, train_labels)
    test_images, test_labels = shuffle_dataset(test_images, test_labels)

    print("Train images dataset shape: " + str(train_images.shape))
    print("Train labels dataset shape: " + str(train_labels.shape))
    print("Test images dataset shape: " + str(test_images.shape))
    print("Test labels dataset shape: " + str(test_labels.shape))

    return train_images, train_labels, test_images, test_labels


def new_prepare_data(path_to_images: str, images_duplication: dict):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for image_class in classes:
        images_paths = os.listdir(os.path.join(path_to_images, image_class))
        # Limit on the number of images
        images_paths = images_paths[:images_restrictions]
        # List with all class images
        images = []
        print('Start prepare %s. Done: %s/%s' % (image_class, classes.index(image_class), len(classes)))

        for num_image, img_name in enumerate(images_paths):
            try:
                image = cv2.imread(os.path.join(path_to_images, image_class, img_name))
                image = cv2.resize(image, images_shape[:2])[:, :, ::-1]

                images.append(image)

                if images_duplication[image_class] > 1:
                    # Horizontal flip
                    images.append(cv2.flip(image, 1))

                if images_duplication[image_class] > 2:
                    # Converting image to LAB Color model
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    # Splitting the LAB image to different channels
                    l, a, b = cv2.split(lab)
                    # Applying CLAHE to L-channel
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
                    # Merge the CLAHE enhanced L-channel with the a and b channel
                    limg = cv2.merge((clahe, a, b))
                    # Converting image from LAB Color model to RGB model
                    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

                    images.append(image)

                printProgressBar(num_image + 1, len(images_paths), prefix='Load images progress:', suffix='Complete')
            except Exception as e:
                print("GET EXCEPTION: " + str(e))

        # Generate lables with the same lenght as images
        labels = [classes.index(image_class)] * len(images)
        print("Load images: " + str(len(images)) + ", load labels: " + str(len(labels)))

        if shuffle_images:
            print('Suffle images')
            np.random.shuffle(images)

        num_to_train = int(len(images) * (1 - percent_to_test))
        train_images += images[:num_to_train]
        train_labels += labels[:num_to_train]
        test_images += images[num_to_train:]
        test_labels += labels[num_to_train:]

    assert len(train_labels) == len(train_images)
    assert len(test_labels) == len(test_images)

    train_images, train_labels = shuffle_dataset(train_images, train_labels)
    test_images, test_labels = shuffle_dataset(test_images, test_labels)

    print("Train images dataset shape: " + str(train_images.shape))
    print("Train labels dataset shape: " + str(train_labels.shape))
    print("Test images dataset shape: " + str(test_images.shape))
    print("Test labels dataset shape: " + str(test_labels.shape))

    return train_images, train_labels, test_images, test_labels


def add_edges_to_data(data, dt: int = None, min_val: int = 100, max_val: int = 200):
    # If dt not specified correctly, add edges to both types
    if dt not in [0, 1]:
        data = add_edges_to_data(data, dt=0, min_val=min_val, max_val=max_val)
        data = add_edges_to_data(data, dt=1, min_val=min_val, max_val=max_val)
    else:
        len_data = len(data[2 * dt])

        for i in range(len_data):
            img = data[2 * dt][i]
            img = get_image_with_edges(img, min_val, max_val)
            data[2 * dt][i] = img

            printProgressBar(i + 1, len_data, prefix='Add edges progress:', suffix='Complete')

    return data


def add_conv2d(model, filters, kernel_size, pool_size=(2, 2), dropout=0.25, input_shape=None):
    if input_shape:
        model.add(
            Conv2D(
                filters, kernel_size, padding="same",
                kernel_regularizer=l2(l2_regularizer_coef),
                activation='relu', input_shape=input_shape
            )
        )
    else:
        model.add(
            Conv2D(
                filters, kernel_size, padding="same",
                kernel_regularizer=l2(l2_regularizer_coef),
                activation='relu'
            )
        )

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, strides=2))

    if dropout:
        model.add(Dropout(dropout))

    return model


def add_dense(model, neurons, dropout=0.5):
    model.add(
        Dense(neurons, kernel_regularizer=l2(l2_regularizer_coef), activation='relu')
    )
    if dropout:
        model.add(Dropout(dropout))

    return model


# Initialize new Keras Model


def init_model(shape, conv2d_num=5, start_filters_num=8, kernel_size=3, start_neurons_num=256):
    model = keras.Sequential()

    for layer_num in range(conv2d_num):
        print("Add %s conv2d layer" % (layer_num + 1))
        model = add_conv2d(
            model,
            filters=start_filters_num * 1 << layer_num,
            kernel_size=kernel_size,
            input_shape=shape if not layer_num else None
        )

    model.add(Flatten())

    model = add_dense(model, neurons=start_neurons_num)
    model = add_dense(model, neurons=start_neurons_num // 4)

    model.add(Dense(len(classes), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train(data=None, epochs=50, batch_size=128, old_model=None, model_path=None, output_dir='./', **kwargs):
    '''
        :param data: (optional, default: None)
            tuple or list with lenght 4. (Train Data, Train Labels, Test Data, Test Labels)
        :param epochs: (optional, default: 50)
            Number of epochs to train model
        :param batch_size: (optional, default: 128)
            Batch size to train model
        :param old_model: (optional, default: None)
            Trained Sequential model to train more
        :param model_path: (optional, default: None)
            Path to model .h5 to train it
        :param output_dir: (optional, default: './')
            Path to directory where to save model .h5 and tensorboard logs
    '''
    # Add tensorboard output
    model_name = 'new_model' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'
    # Path to save model
    new_model_path = os.path.join(output_dir, model_name)
    # Path to save tensorboard logs
    log_dir = os.path.join(output_dir, "logs/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    if data is not None:
        if len(data) == 4:
            train_images, train_labels, test_images, test_labels = data[0], data[1], data[2], data[3]
        else:
            raise Exception("Data is invalid. Check def params to get to know about data param")
    else:
        print("Generate data")
        train_images, train_labels, test_images, test_labels = new_prepare_data(path_to_images=args.images_path)

    if old_model is None and model_path is None:
        # Init new model
        model = init_model(images_shape)
    elif model_path:
        # Load model from model_path
        model = keras.models.load_model(model_path)
    else:
        # Use old model
        model = old_model

    print("Model Summary:\n", model.summary())

    print("Start training")
    model_info = model.fit(
        train_images, train_labels,
        epochs=epochs,
        callbacks=[tensorboard_callback],
        batch_size=batch_size,
        validation_data=(test_images, test_labels),
    )

    print('Save model to ' + new_model_path)
    model.save(new_model_path)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(("Test Loss: %.4f" % test_loss).replace(".", ","))
    print(("Test accuracy: %.4f" % test_acc).replace(".", ","))

    return model, (model_name, model_info)


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', '-i', type=str, default=os.getenv("IMAGES_PATH"),
                        help='Path to dirs with images on every class')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs in training')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size to train model')
    parser.add_argument('--output_dir', '-o', default='./',
                        help='Path to directory where to save model .h5 and tensorboard logs')
    parser.add_argument('--model', '-m', default=None, help='Path to you trained model .h5')
    args = parser.parse_args()

    logging.info("Training preparation")

    if args.images_path is None:
        raise Exception("Images path not given")

    classes = os.listdir(args.images_path)
    logging.info('Classes materials: ' + str(classes))
    # Check data installation and correct
    images_duplication = check_data(args.images_path)
    # Get all data
    all_data = new_prepare_data(args.images_path, images_duplication)

    # Add edges to images
    all_data = add_edges_to_data(all_data, min_val=50, max_val=150)

    # Create model
    initial_model = init_model(images_shape, conv2d_num=5, start_filters_num=8, start_neurons_num=256)

    logging.info('Train artist helper model v.0.0.3')

    # Set timer to check GPU usage
    # thr.Timer(50, check_gpu_work).start()

    # Train model
    test_model, new_model = train(data=all_data, old_model=initial_model, **vars(args))

    MODELS.append(new_model)

    logging.info("Visualization trained model info")

    visual = VisualizationUtils(model=test_model, data=all_data, classes=classes)

    # Visualization functions
    visual.create_images_plot(dt=0)
    visual.create_images_plot(dt=1)

    visual.plot_history(MODELS)
    visual.plot_history(MODELS, key='loss')

    visual.create_table()

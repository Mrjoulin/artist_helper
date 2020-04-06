from pymongo import MongoClient
import logging
import os

MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = int(os.getenv("MONGO_PORT"))
MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME")
MONGO_IMAGES_COLLECTION_NAME = os.getenv("MONGO_IMAGES_COLLECTION_NAME")

logging.info("Connecting to mongodb client: %s:%s" % (MONGO_HOST, MONGO_PORT))
client = MongoClient(MONGO_HOST, MONGO_PORT)

logging.info("Get database: %s" % MONGO_DATABASE_NAME)
db = client[MONGO_DATABASE_NAME]


def new_image(image_id, image_base64):
    '''
    :param image_base64:
    :param image_id:
    :return: None
    '''

    logging.info("Get collection: %s" % MONGO_IMAGES_COLLECTION_NAME)
    col = db[MONGO_IMAGES_COLLECTION_NAME]

    post = {
        "_id": image_id,
        "data": image_base64,
        "class": None
    }

    try:
        col.insert_one(post)
    except Exception as e:
        logging.error("Exception when insert data: " + str(e))

    return None


def update_answer(image_id, answer):
    '''
    :param image_id:
    :param answer:
    :return: None
    '''

    logging.info("Get collection: %s" % MONGO_IMAGES_COLLECTION_NAME)
    col = db[MONGO_IMAGES_COLLECTION_NAME]

    try:
        col.update_one({'_id': image_id}, {"$set": {"class": answer}})
    except Exception as e:
        logging.error("Exception when update answer for image %s: %s. Answer: %s" % (image_id, str(e), answer))

    return None

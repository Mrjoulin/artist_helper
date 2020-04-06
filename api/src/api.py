from flask import Flask, request, jsonify, make_response
from skimage.io import imread
import logging
import base64
import os

from .run import Model
from .db import *

logging.basicConfig(
    format='[%(asctime)s: %(filename)s:%(lineno)s - %(funcName)10s()]%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

app = Flask(__name__)
logging.info('Create NN Model')
model = Model()
PATH_TO_SAVE_IMAGES = os.getenv("PATH_TO_SAVE_IMAGES")

logging.info("Start app")


def make_api_response(payload, code=200):
    success = code == 200
    return make_response(jsonify({'success': success, 'payload': payload}), code)


@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    '''
    @:param:
        Json with:
            "id": <int, unique digit for image>
            "image": "<base64 encoded image from user>"
    @:return:
        Json with:
            "success": <success of request (bool)>,
            "payload" : {
                "predictions": [<array of integers with predictions from neural network (in sum gives 100 %)>],
                "names": ["<array of names detected class material>"]
            }
    '''

    if request.is_json:
        info = request.get_json()
    else:
        return make_api_response({}, code=405)

    if 'image' in info and isinstance(info['image'], str) and \
            'id' in info and isinstance(info['id'], int):
        image_id = info['id']
        image_base64 = info['image']
    else:
        return make_api_response({}, code=405)

    logging.info("Save image to database")
    new_image(
        image_id=image_id,
        image_base64=image_base64
    )

    logging.info('Get image from user. Decode from base64.')
    decode_img = base64.decodebytes(image_base64.split(',')[1].encode('utf-8'))
    path = PATH_TO_SAVE_IMAGES + 'dont_know/image_%s.jpg' % image_id
    with open(path, 'wb') as write_file:
        write_file.write(decode_img)
    image = imread(path)
    os.remove(path)

    logging.info('Start prediction image')
    predict = model.prediction(image)
    result_predictions = []

    for predict_class in predict['prediction']:
        result_predictions.append(int(round(predict_class)))

    result_predictions[0] = 100 - sum(result_predictions[1:])

    # if rounding gives logical mistakes
    if result_predictions[0] < result_predictions[1]:
        result_predictions[0], result_predictions[1] = result_predictions[1], result_predictions[0]

    logging.info('Return: predictions - ' + str(result_predictions) + '; names - ' + str(predict['names']))
    return make_api_response({"predictions": result_predictions, "names": predict['names']})


@app.route('/get_answer', methods=['POST'])
def get_answer():
    '''
        @:param:
            Json with:
                "id": <int, unique digit for image>
                "answer": "<name of class material>"
        '''

    if request.is_json:
        info = request.get_json()
    else:
        return make_api_response({}, code=405)

    if 'answer' in info and isinstance(info['answer'], str) and \
            'id' in info and isinstance(info['id'], int):
        image_id = info['id']
        answer_class = info['answer']
    else:
        return make_api_response({}, code=405)

    update_answer(
        image_id=image_id,
        answer=answer_class
    )

    return make_api_response({})

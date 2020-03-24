from flask import Flask, request, jsonify, make_response, render_template

import logging
from skimage.io import imread
import base64
import os

from backend.run import *

app = Flask(__name__)
model = Model()
PATH_TO_SAVE_IMAGES = './backend/images/get_from_user/'


def make_api_response(payload, code=200):
    success = code == 200
    return make_response(jsonify({'success': success, 'payload': payload}), code)


@app.route("/")
def main():
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def process():
    '''
    @:param:
        Json with:
            "id": <int, unique digit for image>
            "image": "<base64 encoded image from user>"
    @:return:
        {
            "success": <success of request (bool)>,
            "payload" : {
                "predictions": [<array of integers with predictions from neural network (in sum gives 100 %)>],
                "names": ["<array of names detected class material>"]
            }
        }
    '''

    if request.is_json:
        json = request.get_json()
    else:
        return make_api_response({}, code=405)

    if 'image' in json and isinstance(json['image'], str) and \
            'id' in json and isinstance(json['id'], int):
        image_id = json['id']
        image_base64 = json['image']
    else:
        return make_api_response({}, code=405)

    logging.info('Get image from user. Decode from base64.')
    decode_img = base64.b64decode(image_base64.split(',')[1].encode('utf-8'))
    path = PATH_TO_SAVE_IMAGES + 'dont_know/image_%s.jpg' % image_id
    with open(path, 'wb') as write_file:
        write_file.write(decode_img)
    image = imread(path)

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


@app.route('/answer', methods=['POST'])
def answer():
    '''
        @:param:
            Json with:
                "id": <int, unique digit for image>
                "answer": "<name of class material>"
        '''

    if request.is_json:
        json = request.get_json()
    else:
        return make_api_response({}, code=405)

    if 'answer' in json and isinstance(json['answer'], str) and \
            'id' in json and isinstance(json['id'], int):
        image_id = json['id']
        answer_class = json['answer']
    else:
        return make_api_response({}, code=405)

    path_to_save = os.path.join(PATH_TO_SAVE_IMAGES, answer_class)
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    try:
        image_path = PATH_TO_SAVE_IMAGES + 'dont_know/image_%s.jpg' % image_id
        os.rename(image_path, os.path.join(path_to_save, 'image_%s.jpg' % image_id))
    except Exception as e:
        logging.info('GET EXCEPTION: ' + str(e))
        return make_api_response({}, code=405)

    return make_api_response({})


def run_app(host, port, cert_file, key_file):
    app.run(host=host, port=port)

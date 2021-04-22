from flask import Flask, request, jsonify, make_response, render_template
import argparse
import requests
import logging
import os

logging.basicConfig(
    format='[%(asctime)s: %(filename)s:%(lineno)s - %(funcName)10s()]%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

app = Flask(__name__)

SERVER_IP = os.getenv("SERVER_IP") or "35.204.187.136"
SERVER_PORT = os.getenv("SERVER_PORT") or 8000


def make_api_response(payload, code=200):
    success = code == 200

    return make_response(jsonify({'success': success, 'payload': payload}), code)


def make_api_request(method_name, **kwargs):
    url = 'http://api:{port}/{method}'.format(
        port=SERVER_PORT,
        method=method_name
    )

    try:
        response = requests.post(url, json=kwargs).json()
    except Exception as e:
        logging.error(
            "Got Exception: {exception}\nTry to connect to server {server_ip}".format(
                exception=str(e),
                server_ip=SERVER_IP
            )
        )

        url = url.replace('api', SERVER_IP)

        response = requests.post(url, json=kwargs).json()

    logging.debug(str(response))

    if not response['success']:
        raise Exception(response['payload']['message'])

    return response


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
        Response from api (json)
        See more information in api/api.py
    '''

    if request.is_json:
        json = request.get_json()
    else:
        return make_api_response({}, code=405)

    if 'image' in json and isinstance(json['image'], str) and \
            'id' in json and isinstance(json['id'], int):
        logging.info('Create request to get prediction on image')
        response = make_api_request(
            "get_prediction",
            image=json["image"],
            id=json["id"]
        )
    else:
        return make_api_response({}, code=405)

    return make_api_response(response['payload'])


@app.route('/answer', methods=['POST'])
def answer():
    '''
        @:param:
            Json with:
                "id": <int, unique digit for image>
                "answer": "<name of class material>"
        @:return
            Response from api (json)
            See more information in api/api.py
    '''

    if request.is_json:
        json = request.get_json()
    else:
        return make_api_response({}, code=405)

    if 'answer' in json and isinstance(json['answer'], str) and \
            'id' in json and isinstance(json['id'], int):
        response = make_api_request(
            "get_answer",
            answer=json["answer"],
            id=json["id"]
        )
    else:
        return make_api_response({}, code=405)

    return make_api_response(response['payload'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Server options')
    parser.add_argument("--host", default='0.0.0.0', help="Host server")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port server (default: 5000)")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)

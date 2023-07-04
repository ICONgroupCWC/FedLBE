import io

import torch
from flask import Flask, send_file, make_response, request
from Model import MLP_Net
from flask_cors import CORS, cross_origin
from DBService import db_service

app = Flask(__name__)
cors = CORS(app)


@app.route("/receive_weights", methods=['POST'])
def receive_weights():
    print('request received' + str(request.get_json()))
    data = request.get_json()
    user_name = data['user_name']
    task_name = data['task_name']

    model_weights_bytes = db_service.get_model(user_name, task_name)
    model = MLP_Net()
    model.load_state_dict(torch.load(io.BytesIO(model_weights_bytes)))
    # model.load_state_dict(torch.load('E:\RA Work\FedLearningBE\Fed Learning Code\Your Code\model.pt'))
    model_weights = model.state_dict()
    to_send = io.BytesIO()
    torch.save(model_weights, to_send, _use_new_zipfile_serialization=False)
    to_send.seek(0)
    return send_file(to_send, mimetype='application/octet-stream')


@app.route("/receive_data", methods=['POST'])
def receive_data():
    print('request received' + str(request.get_json()))
    data = request.get_json()
    user_name = data['user_name']
    task_name = data['task_name']

    results = db_service.get_data(user_name, task_name)

    return results


@app.route("/receive_tasks", methods=['POST'])
def receive_tasks():
    print('request received' + str(request.get_json()))
    data = request.get_json()
    user_name = data['user_name']

    results = db_service.get_tasks(user_name)

    return results

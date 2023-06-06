import io

import torch
from flask import Flask, send_file, make_response
from Model import MLP_Net
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
@app.route("/receive_weights")
def receive_weights():
    print('request received')
    model = MLP_Net()
    model.load_state_dict(torch.load('E:\RA Work\FedLearningBE\Fed Learning Code\Your Code\model.pt'))
    model_weights = model.state_dict()
    to_send = io.BytesIO()
    torch.save(model_weights, to_send, _use_new_zipfile_serialization=False)
    to_send.seek(0)
    return send_file(to_send, mimetype='application/octet-stream')
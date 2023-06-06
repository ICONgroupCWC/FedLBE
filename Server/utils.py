import json
import pickle
from json import JSONEncoder
import base64
import torch
from torch.utils.data import Dataset


def create_message(batch_size, learning_rate, epochs, model,modelParam, transforms,  weights=None):
    data = [batch_size, learning_rate, epochs,  model, modelParam, transforms]
    if weights:
        data.append(weights)

    return pickle.dumps(data)


def create_message_json(batch_size, learning_rate, epochs, idxs, weights=None):
    data = {'batchsize': str(batch_size), 'lr': str(learning_rate), 'epochs': str(epochs), 'idx': str(idxs)}
    if weights:
        data['weights'] = weights

    serialized_data = json.dumps(data)

    return serialized_data


def create_message_results(accuracy, train_loss, test_loss, cur_round, elapsed_time, weights=None):


    data = {'status': 'results', 'accuracy': str(accuracy), 'train_loss': str(train_loss), 'test_loss': str(test_loss),
            "round": str(cur_round), "round_time": str(elapsed_time)}
    if weights:
        data['model'] = base64.b64encode(pickle.dumps(weights)).decode()

    serialized_data = json.dumps(data)

    return serialized_data

import json
import pickle
from json import JSONEncoder
import base64
import torch
from torch.utils.data import Dataset


def create_message(batch_size, learning_rate, epochs, model, modelParam, transforms, weights=None):
    data = [batch_size, learning_rate, epochs, model, modelParam, transforms]
    if weights:
        data.append(weights)

    return pickle.dumps(data)


def create_message_model_check(batch_size, data_ops, model, model_param):
    data = [batch_size, data_ops, model, model_param]

    return pickle.dumps(data)


def create_message_initalize(extractor_file, rep_file, job_id):
    data = [extractor_file, rep_file, job_id]

    return pickle.dumps(data)


def create_message_rep(offset, batch_size, job_id, rep_weights, het_folder):
    data = [offset, batch_size, job_id, rep_weights, het_folder]

    return pickle.dumps(data)


def create_message_ext(job_id):
    data = [job_id]

    return pickle.dumps(data)


def create_message_optimize(model_weights, job_id, optimizer, lr, criterion, offset, end, rep_outpout, folder):
    data = [model_weights, job_id, optimizer, lr, criterion, offset, end, rep_outpout, folder]

    return pickle.dumps(data)


def create_message_ext_results(job_id, z):
    data = [job_id, z]

    return pickle.dumps(data)


def create_message_shuffle(client_folder):
    data = [client_folder]

    return pickle.dumps(data)


def create_message_json(batch_size, learning_rate, epochs, idxs, weights=None):
    data = {'batchsize': str(batch_size), 'lr': str(learning_rate), 'epochs': str(epochs), 'idx': str(idxs)}
    if weights:
        data['weights'] = weights

    serialized_data = json.dumps(data)

    return serialized_data


def create_message_results(accuracy, train_loss, test_loss, cur_round, elapsed_time, tot_bytes, weights=None):
    data = {'status': 'results', 'accuracy': str(accuracy), 'train_loss': str(train_loss), 'test_loss': str(test_loss),
            "round": str(cur_round), "round_time": str(elapsed_time), 'total_bytes': str(tot_bytes)}
    if weights:
        data['model'] = base64.b64encode(pickle.dumps(weights)).decode()

    serialized_data = json.dumps(data)

    return serialized_data


def create_result_dict(accuracy, train_loss, test_loss, cur_round, elapsed_time):
    data = {'accuracy': str(accuracy[-1]), 'train_loss': str(train_loss[-1]), 'test_loss': str(test_loss[-1]),
            "round": str(cur_round), "round_time": str(elapsed_time)}

    return data


def create_dashboard_msg(web_data, server_data):
    data = web_data['jobData']
    data['general']['method'] = "FedL"
    data['general']['algo'] = server_data['algo']
    data['general']['plots'] = [{"x_axis": "commRounds", "y_axis": "testAccuracy"},
                                {"x_axis": "totTimes", "y_axis": "trainLoss"}]

    data['general']['taskOverview'] = ''

    data['scheme'] = {'minibatch': server_data['minibatch']}
    data['scheme']['epoch'] = server_data['epoch']
    data['scheme']['lr'] = server_data['lr']
    data['scheme']['scheduler'] = server_data['scheduler']
    data['scheme']['clientFraction'] = server_data['clientFraction']
    data['scheme']['minibatchtest'] = server_data['minibatchtest']
    data['scheme']['comRounds'] = server_data['comRounds']

    data['modelParam'] = {'optimizer': server_data['optimizer']}
    data['modelParam']['loss'] = server_data['loss']
    data['modelParam']['compress'] = server_data['compress']

    if server_data['compress'] != 'No':
        data['modelParam']['scale'] = 0.1
        data['modelParam']['z_point'] = 0
        data['modelParam']['num_bits'] = 8

    data['preprocessing'] = {'dtype': server_data['dtype']}
    data['preprocessing']['folder'] = server_data['dataset']
    data['preprocessing']['testfolder'] = server_data['dataset']
    data['preprocessing']['normalize'] = False

    data['modelData'] = {'modelOverview': ''}

    #
    #
    #
    # data =  {"general": {"task": "test", "method": "FedL", "algo": "Classification", "host": "localhost",
    #              "clients": [{"client_ip": "localhost:5001"}],
    #              "plots": [{"x_axis": "commRounds", "y_axis": "testAccuracy"},
    #                        {"x_axis": "totTimes", "y_axis": "trainLoss"}]},
    #  "scheme": {"minibatch": "4", "epoch": "2", "lr": "0.0001", "scheduler": "random", "clientFraction": "0.5",
    #             "minibatchtest": "8", "comRounds": "10"},
    #  "modelParam": {"optimizer": "Adam", "loss": "CrossEntropyLoss", "compress": "quantize", "scale": "0.1",
    #                 "z_point": "0", "num_bits": "8"},
    #  "preprocessing": {"dtype": "img", "folder": "mnist", "testfolder": "mnist", "normalize": false}}
    dashboard_data = {'dashboard': data}
    return dashboard_data

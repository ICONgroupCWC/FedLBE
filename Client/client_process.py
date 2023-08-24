import importlib
import inspect
import os
import pickle
import sys
import uuid
from pathlib import Path
import numpy as np
import torch
import copy
from client_update import ClientUpdate
from modelUtil import quantize_tensor, compress_tensor
import time

def load_dataset(folder):
    mnist_data_train = np.load('data/' + str(folder) + '/X.npy')
    mnist_labels = np.load('data/' + str(folder) + '/y.npy')

    return mnist_data_train, mnist_labels


async def process(job_data, websocket):
    global model, results
    quantized_diff_all = []
    info_all = []
    v_all, i_all, s_all = [], [], []
    # Model architecture python file  submitted in the request is written to the local folder
    # and then loaded as a python class in the following section of the code

    job_id = str(uuid.uuid4()).strip('-')
    filename = "./ModelData/" + str(job_id) + '/Model.py'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        f.write(job_data[3])

    path_pyfile = Path(filename)
    sys.path.append(str(path_pyfile.parent))
    mod_path = str(path_pyfile).replace(os.path.sep, '.').strip('.py')
    imp_path = importlib.import_module(mod_path)

    for name_local in dir(imp_path):

        if inspect.isclass(getattr(imp_path, name_local)):
            modelClass = getattr(imp_path, name_local)
            model = modelClass()

    # Accessing data from the request
    # B = Batchsize
    # eta = Learning rate
    # E = number of local epochs

    B = job_data[0]
    eta = job_data[1]
    E = job_data[2]
    optimizer = job_data[4]['optimizer']
    criterion = job_data[4]['loss']
    compress = job_data[4]['compress']
    dataops = job_data[5]
    global_weights = job_data[-1]
    model.load_state_dict(global_weights)
    torch.save(model.state_dict(), 'model.pt')
    server_model = copy.deepcopy(model)
    ds, labels = load_dataset(dataops['folder'])
    client = ClientUpdate(dataset=ds, batchSize=B, learning_rate=eta, epochs=E, labels=labels, optimizer_type=optimizer,
                          criterion=criterion, dataops=dataops)

    w, l = await client.train(model, websocket)
    model.load_state_dict(w)

    if compress:
        if compress == 'quantize':
            for server_param, client_param in zip(server_model.parameters(), model.parameters()):
                diff = client_param.data - server_param.data

                print('compressing ' + str(compress))
                z_point = float(job_data[4]['z_point'])
                scale = float(job_data[4]['scale'])
                num_bits = int(job_data[4]['num_bits'])
                quantized_diff, info = quantize_tensor(diff, scale, z_point, num_bits=num_bits)
                quantized_diff_all.append(quantized_diff)
                info_all.append(info)
            results = pickle.dumps([quantized_diff_all, l, info_all])
        else:
            for server_param, client_param in zip(server_model.parameters(), model.parameters()):
                diff = client_param.data - server_param.data
                print('compressing ' + str(compress))
                r = float(job_data[4]['r'])
                v, i, s = compress_tensor(diff, r, comp_type=compress)
                v_all.append(v)
                i_all.append(i)
                s_all.append(s)
            results = pickle.dumps([v_all, i_all, s_all, l])

    else:
        results = pickle.dumps([w, l])
    time.sleep(4)
    await websocket.send(results)

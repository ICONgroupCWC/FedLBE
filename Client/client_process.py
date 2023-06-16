import importlib
import inspect
import os
import pickle
import sys
import uuid
from pathlib import Path
import numpy as np
from client_update import ClientUpdate


def load_dataset(folder):

    mnist_data_train = np.load('data/' + str(folder) + '/X.npy')
    mnist_labels = np.load('data/' + str(folder) + '/y.npy')

    return mnist_data_train, mnist_labels


async def process(job_data, websocket):
    global model

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
    dataops = job_data[5]
    global_weights = job_data[-1]
    model.load_state_dict(global_weights)

    ds, labels = load_dataset(dataops['folder'])
    client = ClientUpdate(dataset=ds, batchSize=B, learning_rate=eta, epochs=E, labels=labels, optimizer_type=optimizer,
                          criterion=criterion, dataops=dataops)

    w, l = await client.train(model, websocket)

    results = pickle.dumps([w, l])
    await websocket.send(results)

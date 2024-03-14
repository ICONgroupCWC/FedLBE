import importlib
import inspect
import json
import os
import sys
import uuid
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from DataLoaders.loaderUtil import getDataloader
from modelUtil import get_criterion, get_optimizer


def load_dataset(folder):
    mnist_data_train = np.load('data/' + str(folder) + '/X.npy')
    mnist_labels = np.load('data/' + str(folder) + '/y.npy')

    return mnist_data_train, mnist_labels


async def check_model(job_data, websocket):
    global model

    B = job_data[0]
    dataops = job_data[1]
    optimizer = job_data[3]['optimizer']
    criterion = job_data[3]['loss']



    dataset, labels = load_dataset(dataops['folder'])
    train_loader = DataLoader(getDataloader(dataset, labels, dataops), batch_size=B, shuffle=True)

    job_id = str(uuid.uuid4()).strip('-')
    filename = "./ModelData/" + str(job_id) + '/Model.py'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(job_data[2])

    path_pyfile = Path(filename)
    sys.path.append(str(path_pyfile.parent))
    mod_path = str(path_pyfile).replace(os.path.sep, '.').strip('.py')
    imp_path = importlib.import_module(mod_path)

    for name_local in dir(imp_path):

        if inspect.isclass(getattr(imp_path, name_local)):
            modelClass = getattr(imp_path, name_local)
            model = modelClass()

    data, labels = next(iter(train_loader))

    criterion = get_criterion(criterion)
    optimizer = get_optimizer(optimizer, model, 0.0001)
    try:
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        success_msg = str({"status": "success", "type": "model", "message": "model architecture is valid"})
        print('model passed')
        await websocket.send(success_msg)
    except Exception as e:
        print('model failed ' + str(e))
        err_msg = {"status": "failed", "type": "model", "message": "model architecture not valid"}
        await websocket.send(str(err_msg))

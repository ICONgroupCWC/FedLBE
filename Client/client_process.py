import importlib
import inspect
import os
import pickle
import sys
import uuid
from pathlib import Path
import numpy as np
from client_update import ClientUpdate

def load_dataset():
    # transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_data_train = np.load('data/X.npy')
    mnist_labels = np.load('data/y.npy')
    # mnist_data_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms_mnist)

    return mnist_data_train, mnist_labels


async def process(job_data, websocket):
    ds, labels = load_dataset()
    global model

    job_id = str(uuid.uuid4()).strip('-')
    filename = "./ModelData/" + str(job_id) + '/Model.py'
    print(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        f.write(job_data[4])

    path_pyfile = Path(filename)
    sys.path.append(str(path_pyfile.parent))
    mod_path = str(path_pyfile).replace(os.path.sep, '.').strip('.py')
    imp_path = importlib.import_module(mod_path)

    for name_local in dir(imp_path):

        if inspect.isclass(getattr(imp_path, name_local)):
            print(f'{name_local} is a class')
            modelClass = getattr(imp_path, name_local)
            model = modelClass()
            print(model)

    B = job_data[0]
    eta = job_data[1]
    E = job_data[2]
    ids = job_data[3]
    optimizer = job_data[5]['optimizer']
    criterion = job_data[5]['loss']
    global_weights = job_data[-1]
    model.load_state_dict(global_weights)
    client = ClientUpdate(dataset=ds, batchSize=B, learning_rate=eta, epochs=E, labels=labels,optimizer_type=optimizer, criterion=criterion )
    w, l = await client.train(model, websocket)

    results = pickle.dumps([w, l])
    await websocket.send(results)


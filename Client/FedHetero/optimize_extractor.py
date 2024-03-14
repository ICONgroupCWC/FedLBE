import importlib
import inspect
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

from modelUtil import get_optimizer


def load_dataset(folder):
    data = torch.from_numpy(np.load('data/' + str(folder) + '/X.npy')).to(torch.float32)
    labels = torch.from_numpy(np.load('data/' + str(folder) + '/y.npy')).type(torch.LongTensor)
    return data, labels


async def optimize_extractor(job_data, websocket):
    global own_model

    model_weights = job_data[0]
    job_id = job_data[1]
    optimizer_type = job_data[2]
    lr = job_data[3]
    criterion = job_data[4]
    offset = job_data[5]
    end = job_data[6]
    rep_output = job_data[7]
    folder = job_data[8]

    ext_learner_file = "./ModelData/" + str(job_id) + '/ExtModel.py'
    checkpoint_path = Path("./ModelData/" + str(job_id) + '/checkpoint.pt')


    model_list = []
    path_pyfile_rep = Path(ext_learner_file)
    sys.path.append(str(path_pyfile_rep.parent))
    mod_path = str(path_pyfile_rep).replace(os.path.sep, '.').strip('.py')
    imp_path = importlib.import_module(mod_path)

    for name_local in dir(imp_path):

        if inspect.isclass(getattr(imp_path, name_local)):
            print(f'{name_local} is a class')
            modelClass = getattr(imp_path, name_local)
            for i in range(len(model_weights)):
                model = modelClass()
                model.load_state_dict(model_weights[i])
                model_list.append(model)
            own_model = modelClass()


    outputs = 0
    print('multiplier ' + str((1 / (len(model_weights)))))
    for i in range(len(model_weights)):

        outputs += (1 / (len(model_weights))) * model_list[i](rep_output)

    optimizer = get_optimizer(optimizer_type, own_model, lr)

    if checkpoint_path.is_file():
        print('checkpoint exists')
        path = "./ModelData/" + str(job_id) + '/checkpoint.pt'
        checkpoint = torch.load(path)

        own_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    own_model.train()
    data, labels = load_dataset(folder)
    target = labels[offset:end, :]
    loss = criterion(outputs, torch.squeeze(target))
    loss.backward()
    optimizer.step()
    ckp = {"optimizer_state_dict": deepcopy(optimizer.state_dict()), "model_state_dict": deepcopy(own_model.state_dict())}
    # print('output ' + str(outputs))
    torch.save(ckp, "./ModelData/" + str(job_id) + '/checkpoint.pt')


    message = pickle.dumps({'status': 'done', 'message': 'saved model'})

    await websocket.send(message)

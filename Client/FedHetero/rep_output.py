import importlib
import inspect
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import numpy as np


def load_dataset(folder):
    data = torch.from_numpy(np.load('data/' + str(folder) + '/X.npy')).to(torch.float32)
    labels = torch.from_numpy(np.load('data/' + str(folder) + '/y.npy')).type(torch.float32)
    return data, labels


async def rep_output(job_data, websocket):
    global rep_model
    offset = job_data[0]
    batch_size = job_data[1]
    job_id = job_data[2]
    rep_weights = job_data[3]
    data_folder = job_data[4]

    end = offset + batch_size
    rep_learner_file = "./ModelData/" + str(job_id) + '/RepModel.py'

    path_pyfile_rep = Path(rep_learner_file)
    sys.path.append(str(path_pyfile_rep.parent))
    mod_path = str(path_pyfile_rep).replace(os.path.sep, '.').strip('.py')
    imp_path = importlib.import_module(mod_path)

    for name_local in dir(imp_path):

        if inspect.isclass(getattr(imp_path, name_local)):
            print(f'{name_local} is a class')
            modelClass = getattr(imp_path, name_local)
            rep_model = modelClass()

    rep_model.load_state_dict(rep_weights)

    data, labels = load_dataset(data_folder)

    output = rep_model(data[offset:end, :])
    # print('rep model output' + str(output))
    message = pickle.dumps([output])

    await websocket.send(message)

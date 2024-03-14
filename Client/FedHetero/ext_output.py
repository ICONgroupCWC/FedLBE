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
    labels = torch.from_numpy(np.load('data/' + str(folder) + '/y.npy')).type(torch.LongTensor)
    return data, labels


async def ext_output(job_data, websocket):
    global ext_model
    job_id = job_data[0]

    ext_learner_file = "./ModelData/" + str(job_id) + '/ExtModel.py'
    checkpoint_path = Path("./ModelData/" + str(job_id) + '/checkpoint.pt')

    path_pyfile_rep = Path(ext_learner_file)
    sys.path.append(str(path_pyfile_rep.parent))
    mod_path = str(path_pyfile_rep).replace(os.path.sep, '.').strip('.py')
    imp_path = importlib.import_module(mod_path)

    for name_local in dir(imp_path):

        if inspect.isclass(getattr(imp_path, name_local)):
            print(f'{name_local} is a class')
            modelClass = getattr(imp_path, name_local)
            ext_model = modelClass()

    if checkpoint_path.is_file():
        print('checkpoint exists')
        path = "./ModelData/" + str(job_id) + '/checkpoint.pt'
        checkpoint = torch.load(path)

        ext_model.load_state_dict(checkpoint['model_state_dict'])
    ext_model.eval()
    output = ext_model.state_dict()
    message = pickle.dumps([output])

    await websocket.send(message)

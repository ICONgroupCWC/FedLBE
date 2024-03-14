import json
import os
import pickle
import numpy as np
from sklearn.utils import shuffle
import torch


async def shuffle_dataset(job_data, websocket):

    folder = job_data[0]
    data = np.load('data/' + str(folder) + '/X.npy')
    labels = np.load('data/' + str(folder) + '/y.npy')
    data, labels = shuffle(data, labels)
    np.save('data/' + str(folder) + '/X.npy', data)
    np.save('data/' + str(folder) + '/y.npy', labels)
    message = pickle.dumps({'status': 'done', 'message': 'shuffled data'})

    await websocket.send(message)

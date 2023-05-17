import time
import zmq
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Model import MLP_Net
import pickle
from utils import create_message


def load_dataset():
    transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_data_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms_mnist)
    mnist_data_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms_mnist)

    return mnist_data_train, mnist_data_test


def iid_partition(dataset, K):
    num_items_per_client = int(len(dataset) / K)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(K):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict


async def start_job(job_data, websocket):
    print('start job called')
    schemeData = job_data['scheme']
    client_list = job_data['general']['clients']
    T = int(schemeData['comRounds'])
    C = float(schemeData['clientFraction'])
    K = int(len(client_list))
    E = int(schemeData['epoch'])
    eta = float(schemeData['lr'])
    B = int(schemeData['minibatch'])
    B_test = int(schemeData['minibatchtest'])

    mnist_data_train, mnist_data_test = load_dataset()
    iid_dict = iid_partition(mnist_data_train, K)
    model = MLP_Net()
    criterion = nn.CrossEntropyLoss()
    ds = mnist_data_train

    data_dict = iid_dict
    global_weights = model.state_dict()
    train_loss = []
    test_loss = []
    test_accuracy = []
    context = zmq.Context()

    client_ports = [clt['client_ip'] for clt in client_list]

    socket = context.socket(zmq.REQ)

    w, local_loss = [], []
    m = max(int(C * K), 1)

    S_t = np.random.choice(range(K), m, replace=False)

    # clients = [clt['client_ip'] for clt in client_list]
    # print('clients ' + str(clients))
    clients = [client_ports[i] for i in S_t]
    st_count = 0
    clients_progress = {}
    for client in clients:
        print("tcp://localhost:" + str(client))
        client_add = "tcp://" + str(client) + ":5555"
        # socket.connect("tcp://localhost:" + str(client))
        socket.connect(client_add)
        message = "init "
        message_test = create_message('init', B, eta, E, data_dict[S_t[st_count]], global_weights)

        socket.send(message_test)
        time.sleep(2)

        message_rec = socket.recv()

        if str(message_rec, 'utf-8').split(' ')[0] == 'processing':
            clients_progress[str(client)] = {}
            clients_progress[str(client)]['progress'] = 'InProgress'
            clients_progress[str(client)]['epoch'] = 0
            socket.disconnect(client_add)
        print(message_rec)

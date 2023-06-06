import os
import uuid

import websockets
import asyncio
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import numpy as np
from tqdm import tqdm
import importlib
import inspect
from pathlib import Path
import sys

import pickle

from Server.DataLoaders.loaderUtil import getDataloader
from Server.utils import create_message, create_message_results


class JobServer:

    def __init__(self):
        self.num_clients = 0
        self.local_weights = []
        self.local_loss = []

    def load_dataset(self, folder):

        data_test = np.load('data/' + str(folder) + '/X.npy')
        labels = np.load('data/' + str(folder) + '/y.npy')
        return data_test, labels

    def iid_partition(self, dataset, K):
        num_items_per_client = int(len(dataset) / K)
        client_dict = {}
        image_idxs = [i for i in range(len(dataset))]

        for i in range(K):
            client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
            image_idxs = list(set(image_idxs) - client_dict[i])

        return client_dict

    def testing(self, model, preprocessing, bs, criterion):

        dataset, labels = self.load_dataset(preprocessing['folder'])
        test_loss = 0
        correct = 0
        test_loader = DataLoader(getDataloader(dataset, labels, preprocessing), batch_size=bs, shuffle=False)
        model.eval()
        for data, label, label_2 in test_loader:
            # data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct += pred.eq(label_2.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)

        return test_loss, test_accuracy

    async def connector(self, client_uri, data, server_socket):
        async with websockets.connect(client_uri, ping_interval=None, max_size=3000000) as websocket:
            finished = False
            try:
                await websocket.send(data)
                while not finished:
                    async for message in websocket:
                        try:
                            data = pickle.loads(message)
                            print('pickled data')
                            # TODO check if correctly computed
                            self.local_weights.append(copy.deepcopy(data[0]))
                            self.local_loss.append(copy.deepcopy(data[1]))
                            finished = True
                            break

                        except Exception as e:
                            print('exception ' + str(e))
                            print('not pickled data ' + str(message))
                            await server_socket.send(message)

                print('closed')
            except Exception as e:
                print('exception ' + str(e))

    async def start_job(self, data, websocket):

        global model
        print('start job called')
        job_id = str(uuid.uuid4()).strip('-')
        filename = "./ModelData/" + str(job_id) + '/Model.py'
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            f.write(data['file'])

        path_pyfile = Path(filename)
        sys.path.append(str(path_pyfile.parent))
        mod_path = str(path_pyfile).replace('\\', '.').strip('.py')
        imp_path = importlib.import_module(mod_path)

        for name_local in dir(imp_path):

            if inspect.isclass(getattr(imp_path, name_local)):
                print(f'{name_local} is a class')
                modelClass = getattr(imp_path, name_local)
                model = modelClass()
                print(model)
        job_data = data['jobData']
        schemeData = job_data['scheme']
        client_list = job_data['general']['clients']
        T = int(schemeData['comRounds'])
        C = float(schemeData['clientFraction'])
        K = int(len(client_list))
        E = int(schemeData['epoch'])
        eta = float(schemeData['lr'])
        B = int(schemeData['minibatch'])
        B_test = int(schemeData['minibatchtest'])
        preprocessing = job_data['preprocessing']
        model_data = job_data['modelData']['model'][0]

        # data_train, data_test = self.load_dataset()
        # iid_dict = self.iid_partition(data_train, K)

        # with open(filename, "rb") as source_file:
        #     code = compile(source_file.read(), filename, "exec")
        # exec(code, globals, locals)

        # model = MLP_Net()
        criterion = nn.CrossEntropyLoss()
        # ds = data_train
        #
        # data_dict = iid_dict
        global_weights = model.state_dict()
        train_loss = []
        test_loss = []
        test_accuracy = []
        round_times = []
        m = max(int(C * K), 1)

        # S_t = np.random.choice(range(K), m, replace=False)
        # client_ports = [clt for clt in client_list]
        # # clients = [clt['client_ip'] for clt in client_list]
        # # print('clients ' + str(clients))
        # clients = [client_ports[i] for i in S_t]
        # st_count = 0
        # loop = asyncio.get_running_loop()
        # print('clients ' + str(clients))
        # tasks = []
        for curr_round in tqdm(range(1, T + 1)):
            start_time = time.time()
            S_t = np.random.choice(range(K), m, replace=False)
            client_ports = [clt for clt in client_list]
            # clients = [clt['client_ip'] for clt in client_list]
            # print('clients ' + str(clients))
            clients = [client_ports[i] for i in S_t]
            st_count = 0

            print('clients ' + str(clients))
            tasks = []
            for client in clients:
                client_uri = 'ws://' + str(client['client_ip']) + '/process'
                # websockets.connect(client_uri)
                print(client_uri)
                serialized_data = create_message(B, eta, E,  data['file'], job_data['modelParam'],
                                                 preprocessing, global_weights)
                tasks.append(self.connector(client_uri, serialized_data, websocket))
                # loop.create_task(connector(client_uri, serialized_data, websocket))
                st_count += 0
            await asyncio.gather(*tasks)

            weights_avg = copy.deepcopy(self.local_weights[0])
            for k in weights_avg.keys():
                for i in range(1, len(self.local_weights)):
                    weights_avg[k] += self.local_weights[i][k]

                weights_avg[k] = torch.div(weights_avg[k], len(self.local_weights))

            global_weights = weights_avg

            model.load_state_dict(global_weights)
            torch.save(model.state_dict(), '../model.pt')
            loss_avg = sum(self.local_loss) / len(self.local_loss)
            train_loss.append(loss_avg)

            g_loss, g_accuracy = self.testing(model, preprocessing, B_test, criterion)
            # TODO change to a dict?
            test_loss.append(g_loss)
            test_accuracy.append(g_accuracy)
            elapsed_time = round(time.time() - start_time, 2)
            if len(round_times) > 0:
                tot_time = round_times[-1] + elapsed_time
            else:
                tot_time = elapsed_time

            round_times.append(tot_time)
            # if curr_round == T:
            #     serialized_results = create_message_results(test_accuracy, train_loss, test_loss, curr_round, round_times,global_weights)
            # else:
            serialized_results = create_message_results(test_accuracy, train_loss, test_loss, curr_round, round_times)

            await websocket.send(serialized_results)
            print('calculated results for round ' + str(curr_round))

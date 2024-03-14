import os
import uuid
import websockets
import asyncio
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
import importlib
import inspect
from pathlib import Path
import sys
from modelUtil import dequantize_tensor, decompress_tensor
import pickle
from DataLoaders.loaderUtil import getDataloader
from utils import create_message_initalize, create_message_results, create_result_dict, create_message_rep, \
    create_message_optimize, create_message_ext, create_message_ext_results, create_message_shuffle
from modelUtil import get_criterion
from DBService import db_service
from Scheduler import Scheduler
from sklearn.utils import shuffle
import time


class JobServerHetero:

    def __init__(self):
        self.results = []
        self.new_latencies = None
        self.num_clients = 0
        self.local_outputs = []
        self.outputs = []
        self.local_loss = []
        self.rep_output = []
        self.model_weights = []
        self.comp_len = 0

    def load_dataset(self, folder):

        data_test = torch.from_numpy(np.load('data/' + str(folder) + '/X.npy')).to(torch.float32)
        labels = torch.from_numpy(np.load('data/' + str(folder) + '/y.npy')).type(torch.LongTensor)
        return data_test, labels

    def shuffle_dataset(self, data, labels):

        data, labels = shuffle(data, labels)

        return data, labels

    def testing(self, model, preprocessing, bs, criterion):

        dataset, labels = self.load_dataset(preprocessing['folder'])
        test_loss = 0
        correct = 0
        test_loader = DataLoader(getDataloader(dataset, labels, preprocessing), batch_size=bs, shuffle=False)
        model.eval()
        for data, label in test_loader:
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item() * data.size(0)
            if preprocessing['dtype'] != 'One D':
                _, pred = torch.max(output, 1)
                correct += pred.eq(label.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        if preprocessing['dtype'] != 'One D':
            test_accuracy = 100. * correct / len(test_loader.dataset)
        else:
            test_accuracy = 0

        return test_loss, test_accuracy

    async def connector(self, client_uri, data, client_index, task, server_socket):
        """connector function for connecting the server to the clients. This function is called asynchronously to
        1. send process requests to each client
        2. calculate local weights for each client separately"""

        async with websockets.connect(client_uri, ping_interval=None, max_size=3000000) as websocket:
            finished = False
            try:
                await websocket.send(data)
                start = time.time()
                while not finished:
                    async for message in websocket:
                        try:

                            data = pickle.loads(message)
                            # print('task' + str(task))
                            # print('data ' + str(data))
                            if task == 'rep_output':
                                self.rep_output.append(copy.deepcopy(data[0]))
                            elif task == 'local_output':
                                self.model_weights.append(copy.deepcopy(data[0]))
                            elif task == 'results':
                                self.results.append(copy.deepcopy(data[0]))
                                # print('local output ' + str(self.local_outputs))
                            finished = True
                            break

                        except Exception as e:
                            # print('data exception ' + str(message))
                            await server_socket.send(message)

                # print('closed')
            except Exception as e:
                print('exception ' + str(e))

    async def start_job(self, data, websocket):

        global rep_model, ext_modelClass
        global ext_model
        print('start job called')
        job_id = uuid.uuid4().hex

        # TODO move all following to a method
        rep_learner_file = "./ModelData/" + str(job_id) + '/RepModel.py'
        extractor_file = "./ModelData/" + str(job_id) + '/ExtModel.py'

        os.makedirs(os.path.dirname(rep_learner_file), exist_ok=True)
        os.makedirs(os.path.dirname(extractor_file), exist_ok=True)

        with open(rep_learner_file, 'wb') as f:
            f.write(data['rep_file'])

        with open(extractor_file, 'wb') as f:
            f.write(data['ext_file'])

        print('file written')
        path_pyfile_rep = Path(rep_learner_file)
        sys.path.append(str(path_pyfile_rep.parent))
        mod_path = str(path_pyfile_rep).replace('\\', '.').strip('.py')
        imp_path = importlib.import_module(mod_path)

        for name_local in dir(imp_path):

            if inspect.isclass(getattr(imp_path, name_local)):
                print(f'{name_local} is a class')
                modelClass = getattr(imp_path, name_local)
                rep_model = modelClass()

        path_pyfile_ext = Path(extractor_file)
        sys.path.append(str(path_pyfile_ext.parent))
        mod_path = str(path_pyfile_ext).replace('\\', '.').strip('.py')
        imp_path = importlib.import_module(mod_path)

        for name_local in dir(imp_path):

            if inspect.isclass(getattr(imp_path, name_local)):
                print(f'{name_local} is a class')
                ext_modelClass = getattr(imp_path, name_local)

        job_data = data['jobData']
        schemeData = job_data['scheme']
        client_list = job_data['general']['clients']

        C = float(schemeData['clientFraction']) if 'clientFraction' in schemeData else 1
        schemeData['clientFraction'] = C
        K = int(len(client_list))
        E = int(schemeData['comRounds'])
        rep_lr = float(schemeData['rep_lr'])
        ext_lr = float(schemeData['pred_lr'])
        batch_size = int(schemeData['batch_size'])

        preprocessing = job_data['preprocessing']
        client_folder = preprocessing['folder']
        # db_service.save_job_data(job_data, job_id)

        criterion = get_criterion(job_data['modelParam']['loss'])

        rep_weights = copy.deepcopy(rep_model.state_dict())
        train_loss = []
        test_loss = []
        test_accuracy = []
        round_times = []
        total_bytes = []
        # m = max(int(C * K), 1)

        # run for number of communication rounds

        # TODO change here
        num_examples = 1000
        size = (1000, 1)

        client_ports = [clt for clt in client_list]
        clients = [client_ports[i] for i in range(len(client_list))]

        initial_tasks = []
        for client in clients:
            client_uri = 'ws://' + str(client['client_ip']) + '/initialize_hetero'
            # print(client_uri)
            serialized_data = create_message_initalize(data['ext_file'], data['rep_file'], job_id)
            client_index = client_ports.index(client)
            initial_tasks.append(self.connector(client_uri, serialized_data, client_index, 'initialize', websocket))

        await asyncio.gather(*initial_tasks)
        print('hetero initialized')
        optimizer_rep = torch.optim.Adam(rep_model.parameters(), lr=rep_lr)
        rep_data, rep_labels = self.load_dataset(preprocessing['folder'])
        for curr_round in tqdm(range(1, E + 1)):
            start_time = time.time()
            rep_data, rep_labels = self.shuffle_dataset(rep_data, rep_labels)

            clients = [clt for clt in client_list]
            shuffle_tasks = []
            for client in clients:
                client_uri = 'ws://' + str(client['client_ip']) + '/shuffle'
                # print(client_uri)
                serialized_data = create_message_shuffle(client_folder)
                client_index = client_ports.index(client)
                shuffle_tasks.append(
                    self.connector(client_uri, serialized_data, client_index, 'shuffle', websocket))

            await asyncio.gather(*shuffle_tasks)

            shuffle_tasks.clear()

            st_count = 0

            # print('clients ' + str(clients))
            tasks = []
            count = 0
            period = len(clients) + 1
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                curr_model = period - 1 - (count % period)
                print('current model ' + str(curr_model))
                self.rep_output = []

                if curr_model != len(clients):
                    client = clients[curr_model]
                    client_uri = 'ws://' + str(client['client_ip']) + '/get_rep_output'
                    # TODO change het_data to folder from front end
                    serialized_data = create_message_rep(offset, batch_size, job_id, rep_weights, client_folder)
                    client_index = client_ports.index(client)
                    tasks.append(self.connector(client_uri, serialized_data, client_index, 'rep_output', websocket))
                    await asyncio.gather(*tasks)

                else:
                    # print('using rep for output')
                    rep_out = rep_model(rep_data[offset:end, :])
                    self.rep_output.append(rep_out)

                # print('rep output ' + str(self.rep_output[0]))

                tasks.clear()

                ext_tasks = []
                for client in clients:
                    # print('client ' + str(client))
                    # print('curr model ' + str(curr_model))

                    client_uri = 'ws://' + str(client['client_ip']) + '/get_model_weights'
                    # print(client_uri)
                    serialized_data = create_message_ext(job_id)
                    client_index = client_ports.index(client)
                    ext_tasks.append(
                        self.connector(client_uri, serialized_data, client_index, 'local_output', websocket))
                    st_count += 0

                await asyncio.gather(*ext_tasks)
                # self.rep_output.clear()
                ext_tasks.clear()

                # outputs = 0
                # for idx in range(len(self.local_outputs)):
                #     outputs += (1 / len(clients)) * self.local_outputs[idx]
                #
                # # print('outputs ' + str(outputs.shape))
                # self.local_outputs.clear()
                optim_tasks = []
                if curr_model != len(clients):
                    client = clients[curr_model]
                    client_uri = 'ws://' + str(client['client_ip']) + '/optimize_model'
                    # print(client_uri)
                    serialized_data = create_message_optimize(self.model_weights, job_id, 'Adam', ext_lr, criterion,
                                                              offset, end,
                                                              self.rep_output[0], client_folder)
                    client_index = client_ports.index(client)
                    optim_tasks.append(
                        self.connector(client_uri, serialized_data, client_index, 'optimize_model', websocket))
                    await asyncio.gather(*optim_tasks)

                else:
                    # TODO change here to get from front end
                    # print('training rep model')
                    out_puts = 0
                    for i in range(len(self.model_weights)):
                        model = ext_modelClass()
                        model.load_state_dict(self.model_weights[i])
                        output = model(self.rep_output[0])
                        out_puts += (1 / (len(self.model_weights))) * output

                    rep_model.train()

                    target = rep_labels[offset:end, :]

                    loss = criterion(out_puts, torch.squeeze(target))
                    print('loss ' + str(loss))
                    loss.backward()
                    # for param in rep_model.parameters():
                    #     print(param)
                    optimizer_rep.step()
                    # print('training is it')
                    # print(list(rep_model.parameters())[0].grad)
                optim_tasks.clear()
                self.model_weights.clear()
                self.outputs.clear()

                rep_weights = copy.deepcopy(rep_model.state_dict())
                # print(rep_weights)

                self.local_outputs.clear()
                self.results.clear()
                count += 1

            #
            #
            #
            #
            #
            # print('latencies ' + str(self.new_latencies))
            #
            #
            #     global_weights = model.state_dict()
            #
            # else:
            #     print('not compressing')
            #     weights_avg = copy.deepcopy(self.local_weights[0])
            #     for k in weights_avg.keys():
            #         for i in range(1, len(self.local_weights)):
            #             weights_avg[k] += self.local_weights[i][k]
            #
            #         weights_avg[k] = torch.div(weights_avg[k], len(self.local_weights))
            #
            #     global_weights = weights_avg
            # torch.save(model.state_dict(), "./ModelData/" + str(job_id) + '/model.pt')
            # torch.save(model.state_dict(), 'model.pt')
            # model.load_state_dict(global_weights)
            #
            # loss_avg = sum(self.local_loss) / len(self.local_loss)
            # train_loss.append(loss_avg)
            #
            # g_loss, g_accuracy = self.testing(model, preprocessing, B_test, criterion)
            #
            # test_loss.append(g_loss)
            # test_accuracy.append(g_accuracy)

            z = rep_model(rep_data)

            for client in clients:
                client_uri = 'ws://' + str(client['client_ip']) + '/get_results'
                # print(client_uri)
                serialized_data = create_message_ext_results(job_id, z)
                client_index = client_ports.index(client)
                ext_tasks.append(
                    self.connector(client_uri, serialized_data, client_index, 'results', websocket))
                st_count += 0

            await asyncio.gather(*ext_tasks)

            ext_tasks.clear()

            y_out = 0
            print('results length ' + str(len(self.results)))
            for idx in range(len(self.results)):
                y_out += (1 / len(self.results)) * self.results[idx]

            predictions = y_out.argmax(dim=1)  # Assuming y_ contains raw logits
            #         print(predictions.eq(y_in.data.view_as(predictions)).sum().item()/ len(y_in))
            accuracy = predictions.eq(rep_labels.data.view_as(predictions)).sum().item() / len(rep_labels)
            test_accuracy.append(accuracy)
            elapsed_time = round(time.time() - start_time, 2)
            if len(round_times) > 0:
                tot_time = round_times[-1] + elapsed_time
            else:
                tot_time = elapsed_time

            round_times.append(tot_time)

            t_loss = criterion(y_out, torch.squeeze(rep_labels))
            train_loss.append(t_loss.item())
            print('t loss ' + str(t_loss.item()))
            # if len(total_bytes) > 0:
            #     tot_bytes = total_bytes[-1] + self.bytes[-1] / 1e6
            # else:
            #     tot_bytes = self.bytes[-1] / 1e6
            # total_bytes.append(round(tot_bytes, 2))
            #
            serialized_results = create_message_results(test_accuracy, train_loss, test_loss, curr_round, round_times,
                                                        total_bytes)
            # result_dict = create_result_dict(test_accuracy, train_loss, test_loss, curr_round, elapsed_time)
            # db_service.save_results(result_dict, job_id)
            await websocket.send(serialized_results)
            #
            # print('calculated results for round ' + str(curr_round))

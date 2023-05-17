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


class JobServer:


    def load_dataset(self):
        transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_data_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms_mnist)
        mnist_data_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms_mnist)

        return mnist_data_train, mnist_data_test

    def iid_partition(self, dataset, K):
        num_items_per_client = int(len(dataset) / K)
        client_dict = {}
        image_idxs = [i for i in range(len(dataset))]

        for i in range(K):
            client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
            image_idxs = list(set(image_idxs) - client_dict[i])

        return client_dict

    def testing(self, model, dataset, bs, criterion):
        test_loss = 0
        correct = 0
        test_loader = DataLoader(dataset, batch_size=bs)
        model.eval()
        for data, labels in test_loader:
            # data, labels = data.cuda(), labels.cuda()
            output = model(data)
            loss = criterion(output, labels)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct += pred.eq(labels.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)

        return test_loss, test_accuracy

    # params = {}
    # with open('parameters.txt', 'r') as fr:
    #     lines = fr.readlines()
    #     for line in lines:
    #         print(line.replace(' ', '').strip('\n').split('='))
    #         param = line.replace(' ', '').strip('\n').split('=')
    #         params[param[0]] = param[1]
    # print(params)

    def start_job(self, job_data):
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

        mnist_data_train, mnist_data_test = self.load_dataset()
        iid_dict = self.iid_partition(mnist_data_train, K)
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

        for curr_round in tqdm(range(1, T + 1)):

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

            done = False
            while not done:

                for client in clients:
                    if clients_progress[str(client)]['progress'] == 'InProgress' and clients_progress[str(client)]['epoch'] < E:
                        # print('in if')
                        client_add = "tcp://" + str(client) + ":5555"
                        socket.connect(client_add)
                        socket.send(b'progress')
                        # print('message sent')
                        message_rec = socket.recv()
                        # print('messager recvd in progress')
                        epochs = int(str(message_rec, 'utf-8').split(' ')[-1])
                        # print('epochs ' + str(epochs))
                        clients_progress[str(client)]['epoch'] = epochs

                        if epochs == E:
                            clients_progress[str(client)]['progress'] = 'Done'
                        socket.disconnect(client_add)
                    # print('done')
                    print(clients_progress)

                if not any(clients_progress[key]['progress'] == 'InProgress' for key in clients_progress):
                    done = True
                time.sleep(2)
            time.sleep(1)

            for client in clients:
                # print("tcp://localhost:" + str(client) + ' ' + str(count))
                client_add = "tcp://" + str(client) + ":5555"
                socket.connect(client_add)
                # print('asking for results')
                socket.send(b'results')
                message_rec = socket.recv()
                result = pickle.loads(message_rec)
                weights = result[0]
                loss = result[1]
                w.append(copy.deepcopy(weights))
                local_loss.append(copy.deepcopy(loss))
                # print('results received')
                socket.disconnect(client_add)

            weights_avg = copy.deepcopy(w[0])
            for k in weights_avg.keys():
                for i in range(1, len(w)):
                    weights_avg[k] += w[i][k]

                weights_avg[k] = torch.div(weights_avg[k], len(w))

            global_weights = weights_avg

            model.load_state_dict(global_weights)
            torch.save(model.state_dict(), 'model.pt')
            loss_avg = sum(local_loss) / len(local_loss)
            train_loss.append(loss_avg)

            g_loss, g_accuracy = self.testing(model, mnist_data_test, B_test, criterion)

            test_loss.append(g_loss)
            test_accuracy.append(g_accuracy)
            print('calculated results')

        # fig, ax = plt.subplots()
        # x_axis = np.arange(1, T+1)
        # y_axis = np.array(train_loss)
        # ax.plot(x_axis, y_axis, label = "train loss")
        #
        # y_axis = np.array(test_loss)
        # ax.plot(x_axis, y_axis, label = "test loss")
        #
        # ax.set(xlabel='Number of Rounds', ylabel='Train Loss')
        # ax.legend()
        # ax.grid()
        #
        # plt.show()

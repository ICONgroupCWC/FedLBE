import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Server.CustomDataset import CustomDataset


class ClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs):
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, model, progress):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.5)

        e_loss = []
        for epoch in range(1, self.epochs + 1):
            print('epoch ' + str(epoch))

            train_loss = 0
            model.train()
            for data, labels in self.train_loader:
                # data, labels = data.cuda(), labels.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

            progress.value = epoch
        total_loss = sum(e_loss) / len(e_loss)
        time.sleep(0.1)
        return model.state_dict(), total_loss
import torch
import torch.optim as optim

from copy import deepcopy

import numpy as np


class Client:
    def __init__(self, data, targets, device, noise_level=0):
        self.data = data.to(device)
        self.targets = targets.to(device)
        self.device = device
        self.length = len(self.data)
        if noise_level is None:
            noise_level = 0
        self.noise_level = noise_level

    def train(self, serverModel, criterion, E, B, learning_rate, momentum):
        """
        serverModel - server model
        criterion - loss function (model, data, targets)
        E - number of epochs
        B - number of batches

        returns clientModel.state_dict() after training
        """
        clientModel = deepcopy(serverModel)
        clientModel = clientModel.to(self.device)
        clientModel.load_state_dict(serverModel.state_dict())
        clientOptimiser = optim.SGD(
            clientModel.parameters(), lr=learning_rate, momentum=momentum
        )

        for epoch in range(E):
            batch_indices = self.split_indices(B)
            for batch in range(B):
                data_batch, targets_batch = self.get_subset(batch_indices[batch])
                clientOptimiser.zero_grad()
                loss = criterion(clientModel, data_batch, targets_batch)
                loss.backward()
                clientOptimiser.step()

        self.model = deepcopy(clientModel)
        if self.noise_level > 0:
            clientModel = self.add_noise(clientModel, self.noise_level)
        return clientModel.state_dict()

    def add_noise(self, model, noise_level):
        """
        add noise to all model parameters with given noise_level
        """
        model_state = model.state_dict()
        for key in model_state.keys():
            model_state[key] += torch.normal(
                mean=torch.zeros_like(model_state[key]), std=noise_level
            )
        model.load_state_dict(model_state)
        return model

    def loss(self, model, criterion):
        """
        criterion - loss function (model, data, targets)
        """
        model.eval()
        with torch.no_grad():
            loss = criterion(model, self.data, self.targets)
        model.train()
        return float(loss.cpu())

    def accuracy(self):
        return self.accuracy_(self.model)

    def accuracy_(self, model):
        """
        evaluate model accuracy on client's training data
        """
        model.eval()
        with torch.no_grad():
            scores = model(self.data)
            _, predictions = scores.max(1)
            num_correct = torch.sum(predictions == self.targets)
            total = self.length
            accuracy = num_correct / total
        model.train()
        return float(accuracy.cpu())

    def get_subset(self, indices):
        """
        return a subset of client data and targets with the given indices
        """
        data_raw = [self.data[j] for j in indices]
        targets_raw = [int(self.targets[j]) for j in indices]
        # prepare data and targets for training
        data = torch.stack(data_raw, 0).to(device=self.device).to(torch.float32)
        targets = torch.tensor(targets_raw).to(device=self.device)
        return data, targets

    def split_indices(self, B):
        """
        return a list of indices for B batches
        """
        length = self.length
        indices = list(range(length))
        np.random.shuffle(indices)
        k = int(np.floor(length / B))
        # drops the last few datapoints, if needed, to keep batch size fixed
        if k == 0:
            return [indices for _ in range(B)]  # use all datapoints in each batch
        return [indices[i : i + k] for i in range(0, len(indices), k)]

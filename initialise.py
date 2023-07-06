import torch
import torch.nn as nn
import numpy as np

from data_preprocess import (
    load_mnist_flat,
    load_cifar10,
    NIIDClientSplit,
    synthetic_samples,
)

from model import NN, CNN

from client import Client
from server import Server

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initNetworkData(dataset, num_clients, random_seed, alpha, beta=0):
    """
    choose dataset from ["synthetic", "mnist", "cifar10"]
    num_clients - number of clients
    random_seed - random seed (to ensure consistent client and server data split)
    alpha - Dirichlet parameter (for mnist, cifar10) / Variance (for synthetic)
    beta - Variance parameter (for synthetic only, not needed for mnist, cifar10)
    """
    if dataset not in ["synthetic", "mnist", "cifar10"]:
        raise Exception("Invalid dataset")

    elif dataset == "synthetic":
        clients = []
        test_val_data = []
        test_val_targets = []

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # # distribute data points to num_clients clients by the power law
        # client_datapoint_fractions = np.random.uniform(0, 1, num_clients) ** (
        #     1 / 3
        # )  # inverse CDF sampling
        # distribute datapoints uniformly
        client_datapoint_fractions = np.array([1 for i in range(num_clients)])
        client_datapoint_fractions = client_datapoint_fractions / np.sum(
            client_datapoint_fractions
        )
        total_train_datapoints = 60000
        num_datapoints = total_train_datapoints * client_datapoint_fractions
        for i in range(num_clients):
            N_i = int(num_datapoints[i])
            train_i, test_val_i = synthetic_samples(alpha, beta, N_i)
            clients.append(Client(train_i["data"], train_i["targets"], device))
            test_val_data.extend(test_val_i["data"])
            test_val_targets.extend(test_val_i["targets"])

        serverModel = nn.Sequential(nn.Linear(60, 10))
        # compute total number of datapoints in test_val_data
        test_val_length = len(test_val_data)
        # split these 50:50 between test and val sets
        test_val_indices = list(range(test_val_length))
        np.random.shuffle(test_val_indices)
        test_indices = test_val_indices[: int(test_val_length / 2)]
        val_indices = test_val_indices[int(test_val_length / 2) :]
        test_val_data = torch.stack(test_val_data)
        test_val_targets = torch.stack(test_val_targets)
        val_data = test_val_data[val_indices]
        val_targets = test_val_targets[val_indices]
        test_data = test_val_data[test_indices]
        test_targets = test_val_targets[test_indices]
        server = Server(
            serverModel, val_data, val_targets, test_data, test_targets, device
        )

    elif dataset == "mnist":
        train_dataset, val_dataset, test_dataset = load_mnist_flat()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        client_indices = NIIDClientSplit(train_dataset, num_clients, alpha)
        clients = []
        for i in range(num_clients):
            clients.append(
                Client(
                    train_dataset.data[client_indices[i]],
                    train_dataset.targets[client_indices[i]],
                    device,
                )
            )

        serverModel = NN(input_dim=784, output_dim=10)
        server = Server(
            serverModel,
            val_dataset.data,
            val_dataset.targets,
            test_dataset.data,
            test_dataset.targets,
            device,
        )

    elif dataset == "cifar10":
        train_dataset, val_dataset, test_dataset = load_cifar10()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        client_indices = NIIDClientSplit(train_dataset, num_clients, alpha)
        clients = []
        for i in range(num_clients):
            clients.append(
                Client(
                    train_dataset.data[client_indices[i]],
                    train_dataset.targets[client_indices[i]],
                    device,
                )
            )
        in_channels = 3
        output_dim = 10
        input_h = 32
        input_w = 32
        serverModel = CNN(in_channels, input_w, input_h, output_dim)
        server = Server(
            serverModel,
            val_dataset.data,
            val_dataset.targets,
            test_dataset.data,
            test_dataset.targets,
            device,
        )

    return clients, server

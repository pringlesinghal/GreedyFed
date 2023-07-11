import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

import os
from os.path import exists


def data_transform_images(x):
    to_tensor = transforms.ToTensor()
    x = to_tensor(x)
    x = torch.flatten(x)
    return x


def download_mnist():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    train_data_global = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )  # 60000 samples
    test_val_data_global = datasets.MNIST(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=True
    )  # 10000 samples
    val_data_global, test_data_global = torch.utils.data.random_split(
        test_val_data_global, [0.5, 0.5]
    )

    num_train_points = train_data_global.__len__()
    num_val_points = val_data_global.__len__()
    num_test_points = test_data_global.__len__()

    # random permutation of train indices
    indices = np.random.permutation(num_train_points)
    # shuffle the data
    train_data_global.data = train_data_global.data[indices]
    train_data_global.targets = train_data_global.targets[indices]
    # apply transforms through DataLoader (cheap trick)
    train_dataloader = DataLoader(train_data_global, batch_size=num_train_points)
    train_data_global.data, train_data_global.targets = next(iter(train_dataloader))

    val_dataloader = DataLoader(val_data_global, batch_size=num_val_points)
    val_data_global.data, val_data_global.targets = next(iter(val_dataloader))

    test_dataloader = DataLoader(test_data_global, batch_size=num_test_points)
    test_data_global.data, test_data_global.targets = next(iter(test_dataloader))

    return train_data_global, val_data_global, test_data_global


def load_mnist():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    if not exists("processed_data/mnist/test_data_global.pt"):
        train_data_global, val_data_global, test_data_global = download_mnist()
        os.makedirs("./processed_data/mnist/", exist_ok=True)
        torch.save(train_data_global, "processed_data/mnist/train_data_global.pt")
        torch.save(val_data_global, "processed_data/mnist/val_data_global.pt")
        torch.save(test_data_global, "processed_data/mnist/test_data_global.pt")
    else:
        print("files already downloaded")
        train_data_global = torch.load("processed_data/mnist/train_data_global.pt")
        val_data_global = torch.load("processed_data/mnist/val_data_global.pt")
        test_data_global = torch.load("processed_data/mnist/test_data_global.pt")

    return train_data_global, val_data_global, test_data_global


def download_cifar10():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    train_data_global = datasets.CIFAR10(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )  # 50000 samples
    test_val_data_global = datasets.CIFAR10(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=True
    )  # 10000 samples
    val_data_global, test_data_global = torch.utils.data.random_split(
        test_val_data_global, [0.5, 0.5]
    )

    num_train_points = train_data_global.__len__()
    num_val_points = val_data_global.__len__()
    num_test_points = test_data_global.__len__()

    # apply transforms through DataLoader (cheap trick)

    train_dataloader = DataLoader(train_data_global, batch_size=num_train_points)
    train_data_global.data, train_data_global.targets = next(iter(train_dataloader))
    ### random permutation of train indices
    indices = torch.Tensor(np.random.permutation(num_train_points)).int()
    ### shuffle the data
    train_data_global.data = train_data_global.data[indices]
    train_data_global.targets = train_data_global.targets[indices]

    val_dataloader = DataLoader(val_data_global, batch_size=num_val_points)
    val_data_global.data, val_data_global.targets = next(iter(val_dataloader))

    test_dataloader = DataLoader(test_data_global, batch_size=num_test_points)
    test_data_global.data, test_data_global.targets = next(iter(test_dataloader))

    return train_data_global, val_data_global, test_data_global


def load_cifar10():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    if not exists("processed_data/cifar10/test_data_global.pt"):
        train_data_global, val_data_global, test_data_global = download_cifar10()
        os.makedirs("./processed_data/cifar10/", exist_ok=True)
        torch.save(train_data_global, "processed_data/cifar10/train_data_global.pt")
        torch.save(val_data_global, "processed_data/cifar10/val_data_global.pt")
        torch.save(test_data_global, "processed_data/cifar10/test_data_global.pt")
    else:
        print("files already downloaded")
        train_data_global = torch.load("processed_data/cifar10/train_data_global.pt")
        val_data_global = torch.load("processed_data/cifar10/val_data_global.pt")
        test_data_global = torch.load("processed_data/cifar10/test_data_global.pt")

    return train_data_global, val_data_global, test_data_global


def download_mnist_flat():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    train_data_global = datasets.MNIST(
        root="dataset/", train=True, transform=data_transform_images, download=True
    )  # 60000 samples
    test_val_data_global = datasets.MNIST(
        root="dataset/", train=False, transform=data_transform_images, download=True
    )  # 10000 samples
    val_data_global, test_data_global = torch.utils.data.random_split(
        test_val_data_global, [0.5, 0.5]
    )

    num_train_points = train_data_global.__len__()
    num_val_points = val_data_global.__len__()
    num_test_points = test_data_global.__len__()

    # random permutation of train indices
    indices = np.random.permutation(num_train_points)
    # shuffle the data
    train_data_global.data = train_data_global.data[indices]
    train_data_global.targets = train_data_global.targets[indices]
    # apply transforms through DataLoader (cheap trick)
    train_dataloader = DataLoader(train_data_global, batch_size=num_train_points)
    train_data_global.data, train_data_global.targets = next(iter(train_dataloader))

    val_dataloader = DataLoader(val_data_global, batch_size=num_val_points)
    val_data_global.data, val_data_global.targets = next(iter(val_dataloader))

    test_dataloader = DataLoader(test_data_global, batch_size=num_test_points)
    test_data_global.data, test_data_global.targets = next(iter(test_dataloader))

    return train_data_global, val_data_global, test_data_global


def load_mnist_flat():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    if not exists("processed_data/mnist_flat/test_data_global.pt"):
        train_data_global, val_data_global, test_data_global = download_mnist_flat()
        os.makedirs("./processed_data/mnist_flat/", exist_ok=True)
        torch.save(train_data_global, "processed_data/mnist_flat/train_data_global.pt")
        torch.save(val_data_global, "processed_data/mnist_flat/val_data_global.pt")
        torch.save(test_data_global, "processed_data/mnist_flat/test_data_global.pt")
    else:
        print("files already downloaded")
        train_data_global = torch.load("processed_data/mnist_flat/train_data_global.pt")
        val_data_global = torch.load("processed_data/mnist_flat/val_data_global.pt")
        test_data_global = torch.load("processed_data/mnist_flat/test_data_global.pt")

    return train_data_global, val_data_global, test_data_global


def download_cifar10_flat():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    train_data_global = datasets.CIFAR10(
        root="dataset/", train=True, transform=data_transform_images, download=True
    )  # 50000 samples
    test_val_data_global = datasets.CIFAR10(
        root="dataset/", train=False, transform=data_transform_images, download=True
    )  # 10000 samples
    val_data_global, test_data_global = torch.utils.data.random_split(
        test_val_data_global, [0.5, 0.5]
    )

    num_train_points = train_data_global.__len__()
    num_val_points = val_data_global.__len__()
    num_test_points = test_data_global.__len__()

    # apply transforms through DataLoader (cheap trick)

    train_dataloader = DataLoader(train_data_global, batch_size=num_train_points)
    train_data_global.data, train_data_global.targets = next(iter(train_dataloader))
    ### random permutation of train indices
    indices = torch.Tensor(np.random.permutation(num_train_points)).int()
    ### shuffle the data
    train_data_global.data = train_data_global.data[indices]
    train_data_global.targets = train_data_global.targets[indices]

    val_dataloader = DataLoader(val_data_global, batch_size=num_val_points)
    val_data_global.data, val_data_global.targets = next(iter(val_dataloader))

    test_dataloader = DataLoader(test_data_global, batch_size=num_test_points)
    test_data_global.data, test_data_global.targets = next(iter(test_dataloader))

    return train_data_global, val_data_global, test_data_global


def load_cifar10_flat():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    if not exists("processed_data/cifar10_flat/test_data_global.pt"):
        train_data_global, val_data_global, test_data_global = download_cifar10_flat()
        os.makedirs("./processed_data/cifar10_flat/", exist_ok=True)
        torch.save(
            train_data_global, "processed_data/cifar10_flat/train_data_global.pt"
        )
        torch.save(val_data_global, "processed_data/cifar10_flat/val_data_global.pt")
        torch.save(test_data_global, "processed_data/cifar10_flat/test_data_global.pt")
    else:
        print("files already downloaded")
        train_data_global = torch.load(
            "processed_data/cifar10_flat/train_data_global.pt"
        )
        val_data_global = torch.load("processed_data/cifar10_flat/val_data_global.pt")
        test_data_global = torch.load("processed_data/cifar10_flat/test_data_global.pt")

    return train_data_global, val_data_global, test_data_global


def NIIDClientSplit(train_data, num_clients, alpha):
    """
    train_data: torch.utils.data.Dataset object
    num_clients: int
    alpha: float

    uses dirichlet distribution to distribute data between num_clients clients
    different schemes for distributing data between these clients
    1. diff distribution, same number of points (implemented)
    2. same distribution, diff number of points (not implemented)
    """
    unique_targets, target_counts_global = train_data.targets.unique(return_counts=True)
    num_targets = len(unique_targets)
    target_indices = {int(key): [] for key in unique_targets}
    # first separate data indices by targets
    for idx, target in enumerate(train_data.targets):
        target_indices[int(target)].append(idx)

    # split targets across each client using the dirichlet distribution
    # let client i have labels distributed with proportions [p0-i, p1-i, ..., pM-i] where M is the number of targets
    # larger alpha leads to more uniformness
    alpha = torch.Tensor([alpha])
    target_distribution = Dirichlet(alpha.repeat(num_targets))
    counter = 0
    while True:
        counter += 1
        # number of data points at each client is in proportion given by power law P(x) = 3x^2 (0<x<1)
        # let the total datapoints D be divided as [q0, q1, ..., qN]*D, where N is the number of clients
        # below we sample [q0, q1, ..., qN]
        client_datapoint_fractions = np.random.uniform(0, 1, num_clients) ** (
            1 / 3
        )  # inverse CDF sampling
        client_datapoint_fractions = client_datapoint_fractions / np.sum(
            client_datapoint_fractions
        )
        client_proportions = target_distribution.sample(
            [num_clients]
        )  # sample target distribution for each client

        # client_proportions = (num_clients, num_targets) tensor
        # client_datapoint_fractions = (num_clients) numpy array

        # obtain summation over i (qi pj-i) [for each label j] -> store this in target_fractions[j]
        client_datapoint_fractions = torch.Tensor(client_datapoint_fractions)
        # make -> client_proportions, client_datapoint_fractions = (num_clients, num_targets) tensors
        # and do element-wise multiplication
        target_fractions = client_proportions * torch.t(
            client_datapoint_fractions.repeat(num_targets, 1)
        )
        target_fractions = target_fractions.sum(0)

        # D = min(target_counts[j]/target_fractions[j]) is the maximum number of useful datapoints
        D = torch.min(target_counts_global / target_fractions)
        # D*qi datapoints are allocated to client i
        ## Power Law Distributed number of datapoints
        datapoints_allocated = torch.floor(D * client_datapoint_fractions).int()
        if torch.min(datapoints_allocated) > 30 or counter > 100:
            # to prevent any client from getting too few datapoints
            if counter > 100:
                raise Warning("Unable to allocate sufficient datapoints to each client")
            break

    # now distribute allocated datapoints by Dirichlet distribution
    client_num_datapoints = torch.floor(
        client_proportions * torch.t(datapoints_allocated.repeat(num_targets, 1))
    ).int()

    client_num_datapoints_sum = torch.cumsum(client_num_datapoints, axis=0)
    # shuffle target indices again before assigning them to clients
    for key in target_indices.keys():
        np.random.shuffle(target_indices[key])

    # to store the indices of datapoints that are allocated to each client from train_data
    client_indices = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        for j in range(num_targets):
            lsplit = client_num_datapoints_sum[i - 1][j]
            usplit = client_num_datapoints_sum[i][j]
            if i > 0:
                client_indices[i].extend(target_indices[j][lsplit:usplit])
            else:
                client_indices[i].extend(target_indices[j][0:usplit])

    return client_indices


def DivideIntoBatches(client_indices, num_batches):
    """
    Divides client indices into num_batches batches
    """
    Nb = num_batches
    num_clients = len(client_indices)
    client_indices_batched = [[] for i in range(num_clients)]
    for i in range(num_clients):
        mbs = int(np.floor(len(client_indices[i]) / Nb))  # mini batch size
        # shuffle to ensure different batches each time
        client_indices[i] = np.random.permutation(client_indices[i])
        for j in range(Nb - 1):
            client_indices_batched[i].append(client_indices[i][j * mbs : (j + 1) * mbs])
        client_indices_batched[i].append(
            client_indices[i][(Nb - 1) * mbs :]
        )  # for the last minibatch keep all the remaining points
    return client_indices_batched


def synthetic_samples(alpha, beta, N_i):
    ## create data distribution
    # sample 1 Bk from normal(0, beta)
    B_k = torch.normal(torch.zeros(1), torch.sqrt(torch.ones(1) * beta))
    # sample 60 v from normal(Bk, 1)
    v = torch.normal(B_k, torch.ones(60))
    # construct 60x60 sigma with diag(j,j) = j^(-1.2)
    Sigma = torch.eye(60)
    for i in range(1, 61):
        Sigma[i - 1, i - 1] = i ** (-1.2)
    data_distribution = MultivariateNormal(loc=v, covariance_matrix=Sigma)
    # sample 2*N_i datapoints from data_distribution
    x_train = data_distribution.sample(torch.Size([N_i]))
    x_test = data_distribution.sample(torch.Size([N_i]))

    ## create model distribution
    # sample 1 u_k from normal(0, alpha)
    u_k = torch.normal(torch.zeros(1), torch.sqrt(torch.ones(1) * alpha))
    # sample 10 b_k from normal(u_k, 1)
    b_k = torch.normal(u_k, torch.ones(10))
    # sample 10x60 W_k from normal(u_k, 1)
    W_k = torch.normal(u_k, torch.ones((10, 60)))
    # evaluate targets = argmax(softmax(Wx + b))
    y_train = torch.argmax(torch.matmul(x_train, W_k.t()) + b_k, dim=1)
    y_test = torch.argmax(torch.matmul(x_test, W_k.t()) + b_k, dim=1)

    train = {}
    test_val = {}
    train["data"] = x_train
    train["targets"] = y_train
    test_val["data"] = x_test
    test_val["targets"] = y_test
    return train, test_val

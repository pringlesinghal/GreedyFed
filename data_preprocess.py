import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from os.path import exists
from torch.distributions.dirichlet import Dirichlet


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


def load_mnist():
    """
    returns a tuple of (train_dataset, validation_dataset, test_dataset)
    """
    if not exists("processed_data/mnist/test_data_global.pt"):
        train_data_global, val_data_global, test_data_global = download_mnist()
        torch.save(train_data_global, "processed_data/mnist/train_data_global.pt")
        torch.save(val_data_global, "processed_data/mnist/val_data_global.pt")
        torch.save(test_data_global, "processed_data/mnist/test_data_global.pt")
    else:
        print("files already downloaded")
        train_data_global = torch.load("processed_data/mnist/train_data_global.pt")
        val_data_global = torch.load("processed_data/mnist/val_data_global.pt")
        test_data_global = torch.load("processed_data/mnist/test_data_global.pt")

    return train_data_global, val_data_global, test_data_global


def NIIDClientSplit(train_data, num_clients, alpha):
    """
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

    # larger alpha leads to more central tendency
    alpha = torch.Tensor([alpha])
    target_distribution = Dirichlet(alpha.repeat(num_targets))
    client_proportions = target_distribution.sample(
        [num_clients]
    )  # sample target distribution for each client
    client_num_datapoints = (
        (client_proportions * min(target_counts_global / client_proportions.sum(0)))
        .floor()
        .int()
    )
    client_num_datapoints_sum = np.cumsum(client_num_datapoints, axis=0)

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
        for j in range(Nb - 1):
            client_indices_batched[i].append(client_indices[i][j * mbs : (j + 1) * mbs])
        client_indices_batched[i].append(
            client_indices[i][(Nb - 1) * mbs :]
        )  # for the last minibatch keep all the remaining points
    return client_indices_batched

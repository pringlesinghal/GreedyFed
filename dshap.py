import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import RandomSampler

from tqdm import tqdm
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from os.path import isfile

from data_preprocess import *
from model import NN

"""
the following variables are also defined in main
get rid of them and borrow them from main
"""
input_dim = 784
output_dim = 10
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def DeltaU(x, St, global_dataset):
    """
    x is the index of a single datapoint
    """
    num_epochs = 50
    # create a model
    model = NN(input_dim=input_dim, output_dim=output_dim)
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    # train it on St
    data_raw = [global_dataset.data[i] for i in St]
    targets_raw = [int(global_dataset.targets[i]) for i in St]
    data = torch.stack(data_raw, 0).to(device=device).to(torch.float32)
    targets = torch.tensor(targets_raw).to(device=device)
    for t in range(num_epochs):
        optimiser.zero_grad()
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        scores = model(data)
        loss_init = criterion(scores, targets)

    # train a model on St U x
    model = NN(input_dim=input_dim, output_dim=output_dim)
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    # train it on St
    data_raw.append(global_dataset.data[x])
    targets_raw.append(global_dataset.targets[x])
    data = torch.stack(data_raw, 0).to(device=device).to(torch.float32)
    targets = torch.tensor(targets_raw).to(device=device)
    for t in range(num_epochs):
        optimiser.zero_grad()
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        scores = model(data)
        loss_final = criterion(scores, targets)

    # compute drop in loss
    deltau = loss_init - loss_final
    return float(deltau)


def convergenceTest(dshap):
    """
    Compute average change in last 20 iterations and if it less that 1% it has converged
    """
    if len(dshap) < 20:
        return False
    else:
        return (np.mean(np.abs(dshap[-10:] - dshap[-1])) / dshap[-1]) < 0.01


def DShap(x, S, global_dataset):
    """
    Compute Distributional Shapley on datapoint indices in x using the datapoint indices in S as the database
    typically x = S = client_indices[i]; for some client i
    One-time computation done at the beginning for all client datapoints once client knows about the learning task
    """
    m = len(S)

    # importance sampling for promoting smaller k
    pick_k_from = list(range(1, m))
    wts = np.array([1 / (k + 1) for k in pick_k_from])
    wtsum = np.sum(wts)
    wts = wts / wtsum

    Tmax = 100  # evaluate until Tmax or convergence
    dshap = {i: [0] for i in x}  # to store D-Shap for each index
    for idx, i in enumerate(x):
        t = 1
        Stemp = list(set(S) - set([i]))
        flag = False
        while t <= Tmax:
            k = np.random.choice(pick_k_from, size=1, p=wts)[0]  # draw a random size k
            # then sample a random subset of S of size k
            St_idx = np.random.permutation(len(Stemp))[:k]
            St = [Stemp[i] for i in St_idx]
            deltaU = DeltaU(i, St, global_dataset)
            dshap[i].append((dshap[i][-1] * (t - 1) + deltaU / (m * wts[k - 1])) / t)
            if convergenceTest(dshap[i]):
                print(f"sample {i} converged in {t} steps")
                dshap[i] = dshap[i][-1]  # replace the array with the final value
                flag = True
                break
            t += 1

        if flag == False:
            dshap[i] = dshap[i][-1]
        print(f"D-Shap[{i}] = {dshap[i]}, {len(x) - idx - 1} left")
    return dshap


def DeltaUMiniBatch(x, St, global_dataset, client_indices):
    """
    x is the set of indices in a minibatch for which deltaU is computed
    """
    # validation set
    data_raw_val = [global_dataset.data[i] for i in client_indices]
    targets_raw_val = [int(global_dataset.targets[i]) for i in client_indices]
    data_val = torch.stack(data_raw_val, 0).to(device=device).to(torch.float32)
    targets_val = torch.tensor(targets_raw_val).to(device=device)

    num_epochs = 100
    # create a model
    model = NN(input_dim=input_dim, output_dim=output_dim)
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    # train it on St
    data_raw = [global_dataset.data[i] for i in St]
    targets_raw = [int(global_dataset.targets[i]) for i in St]
    data = torch.stack(data_raw, 0).to(device=device).to(torch.float32)
    targets = torch.tensor(targets_raw).to(device=device)
    for t in range(num_epochs):
        optimiser.zero_grad()
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        scores = model(data_val)
        loss_init = criterion(scores, targets_val)

    # train a model on St U x
    model2 = NN(input_dim=input_dim, output_dim=output_dim)
    model2 = model2.to(device)
    optimiser2 = optim.Adam(model2.parameters(), lr=learning_rate)
    # combine indices in St and x
    indicesunion = list(set(St).union(set(x)))
    data_raw = [global_dataset.data[i] for i in indicesunion]
    targets_raw = [int(global_dataset.targets[i]) for i in indicesunion]
    data = torch.stack(data_raw, 0).to(device=device).to(torch.float32)
    targets = torch.tensor(targets_raw).to(device=device)
    for t in range(num_epochs):
        optimiser2.zero_grad()
        scores = model2(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimiser2.step()

    with torch.no_grad():
        scores = model2(data_val)
        loss_final = criterion(scores, targets_val)

    # compute drop in loss
    deltau = loss_init - loss_final
    # print(f"deltaU  = {deltau} = {loss_init} - {loss_final}")
    return float(deltau)


def DShapMiniBatches(minibatch_indices, S, global_dataset):
    """
    minibatch_indices: list of lists of indices, where each contained list has the indices of datapoints inside a mini-batch
    S: full set of indices for training, typically all datapoints at client side
    global_dataset: set of all training points
    """
    m = len(S)
    # flatten minibatch_indices to get client_indices
    client_indices = []
    for i in minibatch_indices:
        client_indices.extend(i)

    # importance sampling for promoting smaller k
    pick_k_from = list(range(1, m))
    wts = np.array([1 / (k + 1) for k in pick_k_from])
    wtsum = np.sum(wts)
    wts = wts / wtsum

    Tmax = 100  # evaluate until Tmax or convergence
    dshap = {
        idx: [0] for idx, i in enumerate(minibatch_indices)
    }  # to store D-Shap for each index
    for idx, i in enumerate(minibatch_indices):
        t = 1
        Stemp = list(set(S) - set(minibatch_indices[idx]))
        flag = False
        while t <= Tmax:
            k = np.random.choice(pick_k_from, size=1, p=wts)[0]  # draw a random size k
            # then sample a random subset of S of size k
            St_idx = np.random.permutation(len(Stemp))[:k]
            St = [Stemp[i] for i in St_idx]

            deltaU = DeltaUMiniBatch(i, St, global_dataset, client_indices)
            dshap[idx].append(
                (dshap[idx][-1] * (t - 1) + deltaU / (m * wts[k - 1])) / t
            )
            if convergenceTest(dshap[idx]):
                print(f"sample {i} converged in {t} steps")
                dshap[idx] = dshap[idx][-1]  # replace the array with the final value
                flag = True
                break
            t += 1

        if flag == False:
            dshap[idx] = dshap[idx][-1]
        print(f"D-Shap[{idx}] = {dshap[idx]}, {len(minibatch_indices) - idx - 1} left")
    return dshap

import numpy as np
from scipy import stats

import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from main import AlgoResults
from utils import dict_hash

rc("mathtext", default="regular")


def myplot():
    f = plt.figure()
    ax = f.add_subplot(111)
    for idx in range(4):
        val[idx] = np.log(1 + np.multiply(x, a[idx]))  # toy utility model
        plt.plot(
            x, val[idx], color_array[idx] + "-", lw=1.2, label=r"$a={}$".format(a[idx])
        )
    plt.xlabel(r"$\mathbf{x}$", fontsize=13)
    plt.ylabel(r"$\mathbf{v(x)}$", fontsize=13)
    # plt.legend()
    # f.savefig(
    #     "toytex.pdf", bbox_inches="tight", dpi=400
    # )  # comment this for .tex generation
    # plt.show()  # comment this for .tex generation

    # generate .tex
    import tikzplotlib

    tikzplotlib.save("toytex.tex")
    import matplotlib as mpl

    plt.close()
    mpl.rcParams.update(mpl.rcParamsDefault)


# # init
# a = [0.1, 0.2, 0.3, 0.4]
# color_array = ["b", "g", "r", "c"]
# x = np.arange(0, 10, 0.1)
# val = np.zeros((np.size(a), np.size(x)))
# myplot()

config = {
    "algorithm": "ucb",
    "dataset": "synthetic",
    "num_clients": 1000,
    "dataset_alpha": 1,
    "dataset_beta": 1,
    "algo_seed": 0,
    "data_seed": 0,
    "E": 10,
    "B": 10,
    "select_fraction": 0.01,
    "T": 100,
    "lr": 0.01,
    "momentum": 0.5,
    "mu": None,
    "algo_alpha": None,
    "algo_beta": 1,
    "decay_factor": None,
    "noise_level": 0,
}

with open(f"results\{dict_hash(config)}.pickle", "rb") as f:
    results = pickle.load(f)
    test_acc, train_acc, train_loss, val_loss, test_loss = results.get_results()
    results.plot_accuracy()

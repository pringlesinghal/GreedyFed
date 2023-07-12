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


def average_results(results):
    """
    takes input as list of AlgoResults objects
    returns a single AlgoResults object with the average results
    """
    test_acc_avg = []
    train_acc_avg = []
    train_loss_avg = []
    val_loss_avg = []
    test_loss_avg = []

    for run, result in enumerate(results):
        test_acc, train_acc, train_loss, val_loss, test_loss = result.get_results()
        if run == 0:
            test_acc_avg = np.array(test_acc)
            train_acc_avg = np.array(train_acc)
            train_loss_avg = np.array(train_loss)
            val_loss_avg = np.array(val_loss)
            test_loss_avg = np.array(test_loss)
        else:
            test_acc_avg = (run * test_acc_avg + np.array(test_acc)) / (run + 1)
            train_acc_avg = (run * train_acc_avg + np.array(train_acc)) / (run + 1)
            train_loss_avg = (run * train_loss_avg + np.array(train_loss)) / (run + 1)
            val_loss_avg = (run * val_loss_avg + np.array(val_loss)) / (run + 1)
            test_loss_avg = (run * test_loss_avg + np.array(test_loss)) / (run + 1)
    return AlgoResults(
        test_acc_avg, train_acc_avg, train_loss_avg, val_loss_avg, test_loss_avg
    )


rootdir = "results-synthetic11"
# algorithms = ["ucb", "fedprox", "fedavg", "poc", "sfedavg"]
# # select_fractions = [10 / 700, 20 / 700, 30 / 700]
# select_fractions = [10 / 700]
# num_clients = 700
# num_selected_arr = [10, 20, 30]

# results_dict = {}
# for i in algorithms:
#     results_dict[i] = {}
#     for j in num_selected_arr:
#         results_dict[i][j] = []

# sfedavg_alphas = [0, 0.25, 0.5, 0.75]
# poc_decay_factors = [1, 0.9]
# fedprox_mus = [0.001, 0.01, 0.1, 1]
# ucb_betas = [0.01, 0.1, 1, 10, 100]

# # for num_selected = 10
# sfedavg_alpha = 0.5
# poc_decay_factor = 1
# fedprox_mu = 1
# ucb_beta = 100

# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         file_path = os.path.join(subdir, file)
#         with open(file_path, "rb") as f:
#             results = pickle.load(f)
#             algorithm = results.config["algorithm"]
#             select_fraction = results.config["select_fraction"]
#             num_selected = int(select_fraction * num_clients)
#             flag = False
#             if algorithm == "ucb" and results.config["algo_beta"] == ucb_beta:
#                 flag = True
#             elif algorithm == "fedprox" and results.config["mu"] == fedprox_mu:
#                 flag = True
#             elif (
#                 algorithm == "sfedavg" and results.config["algo_alpha"] == sfedavg_alpha
#             ):
#                 flag = True
#             elif (
#                 algorithm == "poc"
#                 and results.config["decay_factor"] == poc_decay_factor
#             ):
#                 flag = True
#             elif algorithm == "fedavg":
#                 flag = True

#             if num_selected != 10:
#                 flag = False

#             if flag:
#                 (results_dict[algorithm][num_selected]).append(results)

# # # for num_selected = 20
# # sfedavg_alpha = 0.5
# # poc_decay_factor = 1
# # fedprox_mu = 0.1
# # ucb_beta = 10

# # for subdir, dirs, files in os.walk(rootdir):
# #     for file in files:
# #         file_path = os.path.join(subdir, file)
# #         with open(file_path, "rb") as f:
# #             results = pickle.load(f)
# #             algorithm = results.config["algorithm"]
# #             select_fraction = results.config["select_fraction"]
# #             num_selected = int(select_fraction * num_clients)
# #             flag = False
# #             if algorithm == "ucb" and results.config["algo_beta"] == ucb_beta:
# #                 flag = True
# #             elif algorithm == "fedprox" and results.config["mu"] == fedprox_mu:
# #                 flag = True
# #             elif (
# #                 algorithm == "sfedavg" and results.config["algo_alpha"] == sfedavg_alpha
# #             ):
# #                 flag = True
# #             elif (
# #                 algorithm == "poc"
# #                 and results.config["decay_factor"] == poc_decay_factor
# #             ):
# #                 flag = True
# #             elif algorithm == "fedavg":
# #                 flag = True

# #             if num_selected != 10:
# #                 flag = False

# #             if flag:
# #                 (results_dict[algorithm][num_selected]).append(results)

# # # for num_selected = 30
# # sfedavg_alpha = 0.5
# # poc_decay_factor = 1
# # fedprox_mu = 0.1
# # ucb_beta = 10

# # for subdir, dirs, files in os.walk(rootdir):
# #     for file in files:
# #         file_path = os.path.join(subdir, file)
# #         with open(file_path, "rb") as f:
# #             results = pickle.load(f)
# #             algorithm = results.config["algorithm"]
# #             select_fraction = results.config["select_fraction"]
# #             num_selected = int(select_fraction * num_clients)
# #             flag = False
# #             if algorithm == "ucb" and results.config["algo_beta"] == ucb_beta:
# #                 flag = True
# #             elif algorithm == "fedprox" and results.config["mu"] == fedprox_mu:
# #                 flag = True
# #             elif (
# #                 algorithm == "sfedavg" and results.config["algo_alpha"] == sfedavg_alpha
# #             ):
# #                 flag = True
# #             elif (
# #                 algorithm == "poc"
# #                 and results.config["decay_factor"] == poc_decay_factor
# #             ):
# #                 flag = True
# #             elif algorithm == "fedavg":
# #                 flag = True

# #             if num_selected != 10:
# #                 flag = False

# #             if flag:
# #                 (results_dict[algorithm][num_selected]).append(results)
# # f = plt.figure()
# # ax = f.add_subplot(111)
# for num_selected in num_selected_arr[0:1]:
#     for algorithm in algorithms:
#         algo_results = results_dict[algorithm][num_selected]
#         summary_results = average_results(algo_results)
#         if algorithm == "ucb":
#             algotext = r"""Fed-Shap-UCB, $\beta = 100$"""
#         elif algorithm == "poc":
#             algotext = r"""Power-Of-Choice"""
#         elif algorithm == "fedavg":
#             algotext = r"""FedAvg"""
#         elif algorithm == "fedprox":
#             algotext = r"FedProx"
#         elif algorithm == "sfedavg":
#             algotext = r"S-FedAvg"

#         plt.plot(summary_results.test_acc, label=algotext)
# plt.ylabel("Test Accuracy")
# plt.xlabel("Communication Rounds")
# # plt.ylim(2, 3)
# plt.legend()
# # plt.show()  # comment this for .tex generation
# # generate .tex
# import tikzplotlib

# tikzplotlib.save(f"plots/test-acc-{rootdir}.tex")
# import matplotlib as mpl

# plt.close()
# mpl.rcParams.update(mpl.rcParamsDefault)

a = []
print(a)
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        file_path = os.path.join(subdir, file)
        with open(file_path, "rb") as f:
            results = pickle.load(f)
            algorithm = results.config["algorithm"]
            dataset = results.config["dataset"]
            num_clients = results.config["num_clients"]
            select_fraction = results.config["select_fraction"]
            num_selected = int(select_fraction * num_clients)
            if num_selected == 30:
                if algorithm == "ucb":
                    x = np.mean(results.num_model_evaluations["gtg"])
                    print(f"x = {x}")
                    a.append(x)
print(a)
print(np.mean(a))

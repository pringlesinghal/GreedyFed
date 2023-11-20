import numpy as np
from scipy import stats
import pandas as pd
import wandb

import pickle
import os
import pprint
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from main import AlgoResults
from utils import dict_hash

import warnings
warnings.filterwarnings("ignore")

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


# Login to your wandb account (only needed once)

# Set your project name and entity (if applicable)
result_path_parent = "final_results/"
dataset = "cifar10"
print(dataset)
if dataset in ["fmnist","mnist"]:
    select_fraction_main = 0.01 # 0.03
    T_main = 400
    T_array = [150, 250, 350]
    # For mnist and fmnist
    select_fraction_main = 0.01
    if dataset in ["fmnist"]:
        GLOBAL_PROJECT_NAME = "FL-FMNIST-Final"
    else:
        GLOBAL_PROJECT_NAME = "FL-MNIST-Final"
elif dataset in ["cifar10"]:
    select_fraction_main = 0.1
    T_main = 200      
    T_array = [100, 150, 200]
    select_fraction_main = 0.1
    GLOBAL_PROJECT_NAME = "FL-CIFAR10-Final"
project_name = GLOBAL_PROJECT_NAME


download_again = False
print(f"download status = {download_again}")
# entity = "your_entity"  # Optional if the project is in your default entity
# Fetch run data from wandb
result_path = os.path.join(result_path_parent, f"{dataset}-runs.pkl")
if not os.path.exists(result_path_parent):
    os.makedirs(result_path_parent)
if os.path.exists(result_path) and download_again == False:
    with open(result_path, "rb") as f:
        runs = pickle.load(f)
else:
    wandb.login()
    runs = wandb.Api().runs(
        f"{project_name}",
        # filters={
        #     "config.noise_level": {"$in": [0, 0.05, 0.1]},
        #     "config.select_fraction": {"$in": [0.01]},
        #     "config.dataset_alpha": {"$in": [1e-4, 1e-1, 1e1]},
        #     "config.systems_heterogenity": {"$in": [0, 0.5, 0.9]},
        # },
    )
    with open(result_path, "wb") as f:
        pickle.dump(runs, f)

result_df_path = os.path.join(result_path_parent, f"{dataset}-runs-df.pkl")
if os.path.exists(result_df_path) and download_again == False:
    with open(result_df_path, "rb") as f:
        result_df_original = pickle.load(f)
else:
    completed_runs = []
    print(len(runs))
    for run in runs:
        config = run.config

        if (
            len(config) > 0
            and config["select_fraction"] == select_fraction_main
        ):
            run_data = run.history(
                keys=[
                    "test_accuracy",
                    "train_accuracy",
                    "train_loss",
                    "val_loss",
                    "test_loss",
                    "_timestamp"
                ]
            )
            if len(run_data) < T_main:
                continue
            run_config = pd.DataFrame.from_dict([config])
            merged_df = (
                run_data.assign(key=1)
                .merge(run_config.assign(key=1), on="key")
                .drop("key", axis=1)
            )
            completed_runs.append(merged_df)
            print(len(completed_runs))

    result_df_original = pd.concat(completed_runs, ignore_index=True)
    with open(result_df_path, "wb") as f:
        pickle.dump(result_df_original, f)
algo_list = ["greedyshap","ucb","sfedavg","fedavg","fedprox","poc","centralised"]
algo_list_formal = ["GreedyFed","UCB","S-FedAvg","FedAvg","FedProx","Power-Of-Choice","Centralized"]
def format_results(T_results, smoothed_df):
    T_results = T_results - 1
    resultsgroup = smoothed_df[smoothed_df["_step"] == T_results].groupby(["algorithm", "memory","mu"], dropna = False)
    accuracy_mean = resultsgroup.mean()["test_accuracy"]
    accuracy_std = resultsgroup.std()["test_accuracy"]
    best_idxs = accuracy_mean.groupby("algorithm").idxmax()
    accuracy_mean = accuracy_mean[best_idxs]
    accuracy_std = accuracy_std[best_idxs]
    # Create a custom categorical data type
    algo_cat = pd.CategoricalDtype(categories=algo_list, ordered=True)
    df = accuracy_mean
    # Apply the custom data type to the "algorithm" column
    df.index = df.index.set_levels(df.index.levels[0].astype(algo_cat), level='algorithm')
    accuracy_mean = list(accuracy_mean.sort_index(level="algorithm"))
    df = accuracy_std
    # Apply the custom data type to the "algorithm" column
    df.index = df.index.set_levels(df.index.levels[0].astype(algo_cat), level='algorithm')
    accuracy_std = list(accuracy_std.sort_index(level="algorithm"))
    result_list = []
    for idx, algorithm in enumerate(algo_list):
        # print(f"{algorithm} ${100*accuracy_mean[idx]:.2f}"+" \pm "+f"{100*accuracy_std[idx]:.2f}$")
        result_list.append(f"${100*accuracy_mean[idx]:.2f} \pm {100*accuracy_std[idx]:.2f}$")
    return result_list
def smoothen_df(alpha, result_df):
    smoothed_df = pd.DataFrame()
    # Smooth the data for each unique combination of 'algorithm' and 'algo_seed' separately using EWMA

    for (algorithm, seed, memory, mu, dataset_alpha, noise, systems_heterogenity), group in result_df.groupby(
        ["algorithm", "seed", "memory","mu","dataset_alpha", "noise", "systems_heterogenity"], dropna = False
    ):
        # keep only the latest run if all parameters are same
        if (group["_timestamp"].head(T_main).values > group["_timestamp"].tail(T_main).values)[0]:
            algorithm_df = group.head(T_main)
        else:
            algorithm_df = group.tail(T_main)
        # Sort the unique step values for interpolation
        sorted_steps = np.sort(algorithm_df["_step"].unique())

        # Perform EWMA smoothing on the accuracy
        smoothed_accuracy = algorithm_df["test_accuracy"].ewm(alpha=alpha).mean()

        algorithm_smoothed_df = pd.DataFrame(
            {
                "_step": sorted_steps,
                "test_accuracy": smoothed_accuracy,
                "algorithm": algorithm,
                "seed": seed,
                "memory": memory,
                "mu":mu,
                "dataset_alpha":dataset_alpha,
                "noise":noise,
                "systems_heterogenity":systems_heterogenity,
            }
        )
        smoothed_df = pd.concat([smoothed_df, algorithm_smoothed_df])
    return smoothed_df

def generate_timing_results(result_df_original, smoothing=0, T = [150, 250, 350]):
    noise = 0
    dataset_alpha = 1e-4
    systems_heterogenity = 0

    df_filter_global = (
        (result_df_original["noise"] == noise)

        & (result_df_original["dataset_alpha"] == dataset_alpha)
        & (result_df_original["systems_heterogenity"] == systems_heterogenity)
        & (result_df_original["algo_beta"] != 0.01)
    )
    result_df = result_df_original.copy(deep=True)
    result_df = result_df[df_filter_global]
    smoothed_df = smoothen_df(1 - smoothing, result_df)

    # Timing constraints data MNIST
    results_1 = format_results(T_results=T[0], smoothed_df=smoothed_df)
    results_2 = format_results(T_results=T[1], smoothed_df=smoothed_df)
    results_3 = format_results(T_results=T[2], smoothed_df=smoothed_df)
    for algo, a, b ,c in zip(algo_list_formal, results_1, results_2, results_3):
        print(f"& {algo} & {a} & {b} & {c} \\\\")

def generate_datahet_results(result_df_original, smoothing=0):
    noise = 0
    systems_heterogenity = 0
    T = T_main

    df_filter_global = (
        (result_df_original["noise"] == noise)
        & (result_df_original["systems_heterogenity"] == systems_heterogenity)
        & (result_df_original["algo_beta"] != 0.01)
    )
    result_df = result_df_original.copy(deep=True)
    result_df = result_df[df_filter_global]
    smoothed_df = smoothen_df(1 - smoothing, result_df)

    # Timing constraints data MNIST
    results_1 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["dataset_alpha"]==1e-4])
    results_2 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["dataset_alpha"]==1e-1])
    results_3 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["dataset_alpha"]==1e1])
    for algo, a, b ,c in zip(algo_list_formal, results_1, results_2, results_3):
        print(f"& {algo} & {a} & {b} & {c} \\\\")

def generate_systems_results(result_df_original, smoothing=0):
    noise = 0
    dataset_alpha = 1e-4
    T = T_main

    df_filter_global = (
        (result_df_original["noise"] == noise)
        & (result_df_original["dataset_alpha"] == dataset_alpha)
        & (result_df_original["algo_beta"] != 0.01)
    )
    result_df = result_df_original.copy(deep=True)
    result_df = result_df[df_filter_global]

    smoothed_df = smoothen_df(1 - smoothing, result_df)

    # Timing constraints data MNIST
    results_1 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["systems_heterogenity"]==0])
    results_2 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["systems_heterogenity"]==0.5])
    results_3 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["systems_heterogenity"]==0.9])
    for algo, a, b ,c in zip(algo_list_formal, results_1, results_2, results_3):
        print(f"& {algo} & {a} & {b} & {c} \\\\")

def generate_noise_results(result_df_original, smoothing=0):
    systems_heterogenity = 0
    dataset_alpha = 1e-4
    T = T_main

    df_filter_global = (
        (result_df_original["systems_heterogenity"] == systems_heterogenity)
        & (result_df_original["dataset_alpha"] == dataset_alpha)
        & (result_df_original["algo_beta"] != 0.01)
    )
    result_df = result_df_original.copy(deep=True)
    result_df = result_df[df_filter_global]

    smoothed_df = smoothen_df(1 - smoothing, result_df)

    # Timing constraints data MNIST
    results_1 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["noise"]==0])
    results_2 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["noise"]==0.05])
    results_3 = format_results(T_results=T, smoothed_df=smoothed_df[smoothed_df["noise"]==0.1])
    for algo, a, b ,c in zip(algo_list_formal, results_1, results_2, results_3):
        print(f"& {algo} & {a} & {b} & {c} \\\\")

print("data")
generate_datahet_results(result_df_original)
print("timing")
generate_timing_results(result_df_original, T=T_array)
print("systems")
generate_systems_results(result_df_original)
print("noise")
generate_noise_results(result_df_original)





# df_filter_ucb = (result_df["algorithm"] == "ucb") & (result_df["algo_beta"] == ucb_beta)
# df_filter_poc = (result_df["algorithm"] == "poc") & (
#     result_df["decay_factor"] == poc_decay_factor
# )
# df_filter_sfedavg = (result_df["algorithm"] == "sfedavg") & (
#     result_df["algo_alpha"] == sfedavg_alpha
# )
# df_filter_fedprox = (result_df["algorithm"] == "fedprox") & (
#     result_df["mu"] == fedprox_mu
# )
# df_filter_fedavg = result_df["algorithm"] == "fedavg"
# df_filter_centralised = result_df["algorithm"] == "centralised"

# df_filter = df_filter_global & (
#     df_filter_fedavg
#     | df_filter_fedprox
#     | df_filter_ucb
#     | df_filter_poc
#     | df_filter_sfedavg
#     | df_filter_centralised
# )
# # print(f"ucb = {(df_filter_ucb & df_filter_global).sum()}")
# # print(f"poc = {(df_filter_poc & df_filter_global).sum()}")
# # print(f"fedprox = {(df_filter_fedprox & df_filter_global).sum()}")
# # print(f"fedavg = {(df_filter_fedavg & df_filter_global).sum()}")
# # print(f"sfedavg = {(df_filter_sfedavg & df_filter_global).sum()}")

# # print(
# #     result_df[
# #         df_filter_fedavg
# #         & df_filter_global
# #         & (result_df["algo_seed"] == 3)
# #         & (result_df["_step"] == 199)
# #     ]
# # )

# result_df = result_df[df_filter]
# # Create a new DataFrame to store the smoothed data
# smoothed_df = pd.DataFrame()

# # Smooth the data for each unique combination of 'algorithm' and 'algo_seed' separately using EWMA
# alpha = (
#     0.2  # The smoothing factor (you can adjust this to control the level of smoothing)
# )

# for (algorithm, algo_seed), group in result_df.groupby(["algorithm", "algo_seed"]):
#     algorithm_df = group
#     if algorithm == "fedavg" and algo_seed == 3:
#         algorithm_df = algorithm_df[0:200]
#     # Sort the unique step values for interpolation
#     sorted_steps = np.sort(algorithm_df["_step"].unique())

#     # Perform EWMA smoothing on the accuracy
#     smoothed_accuracy = algorithm_df["train_accuracy"].ewm(alpha=alpha).mean()

#     algorithm_smoothed_df = pd.DataFrame(
#         {
#             "_step": sorted_steps,
#             "train_accuracy": smoothed_accuracy,
#             "algorithm": algorithm,
#             "algo_seed": algo_seed,
#         }
#     )
#     smoothed_df = pd.concat([smoothed_df, algorithm_smoothed_df])

# sns.set_palette("deep")
# # Create the line plot with seaborn using the smoothed data
# g = sns.lineplot(
#     data=smoothed_df, x="_step", y="test_accuracy", hue="memory", palette="deep"
# )


# plt.ylabel("Training Accuracy")
# plt.xlabel("Communication Rounds")
# plt.ylim([0, 1])
# # fedprox_text = "FedProx, " + r"$\mu = $" + f"{fedprox_mu}"
# # fedavg_text = "FedAvg"
# # ucb_text = "Fed-Shap-UCB, " + r"$\beta = $" + f"{ucb_beta}"
# # poc_text = "Power-Of-Choice, " + r"$\lambda = $" + f"{poc_decay_factor}"
# # sfedavg_text = "S-FedAvg, " + r"$\alpha = $" + f"{sfedavg_alpha}"
# # centralised_text = "Centralised"
# g.legend_.set_title(r"$\beta$")
# # ensure labels are in correct order
# new_labels = [
#     centralised_text,
#     fedavg_text,
#     fedprox_text,
#     poc_text,
#     sfedavg_text,
#     ucb_text,
# ]
# # new_labels = [
# #     fedavg_text,
# #     ucb_text,
# # ]
# for t, l in zip(g.legend_.texts, new_labels):
#     t.set_text(l)
# plt.savefig(
#     f"plots/mnist-noise-ucb-{select_fraction:.3f}-{dataset_alpha}.pdf",
#     format="pdf",
#     bbox_inches="tight",
# )

# plt.show()


# noise_levels = [0, 1e-3, 1e-1]
# dataset_alphas = [1e-3, 1, 1e3]
# algorithms = ["ucb", "fedavg", "fedprox", "sfedavg", "poc"]
# select_fractions = [5 / 300, 25 / 300, 125 / 300]
# sfedavg_alphas = [0.25, 0.5, 0.75]
# poc_decay_factors = [1, 0.9]
# fedprox_mus = [0.001, 0.01, 0.1, 1, 10]
# ucb_betas = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]


# result_data = []

# rootdir = "results-mnist-final"
# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         file_path = os.path.join(subdir, file)
#         with open(file_path, "rb") as f:
#             results = pickle.load(f)
#             algorithm = results.config["algorithm"]
#             select_fraction = results.config["select_fraction"]
#             num_clients = results.config["num_clients"]
#             num_selected = int(select_fraction * num_clients)
#             try:
#                 noise_level = results.config["noise_level"]
#             except KeyError:
#                 noise_level = 0
#             results.config["noise_level"] = noise_level
#             if (
#                 results.config["T"] == 200
#                 and results.config["num_clients"] == 300
#                 and results.config["dataset"] == "mnist"
#             ):
#                 dict_common = deepcopy(results.config)
#                 (
#                     test_acc,
#                     train_acc,
#                     train_loss,
#                     val_loss,
#                     test_loss,
#                 ) = results.get_results()
#                 T = dict_common["T"]
#                 communication_rounds = list(range(T))
#                 plot_vals = {
#                     "rounds": communication_rounds,
#                     "test_acc": test_acc,
#                     "train_acc": train_acc,
#                     "train_loss": train_loss,
#                     "val_loss": val_loss,
#                     "test_loss": test_loss,
#                 }
#                 for i in range(T):
#                     dict_common_copy = deepcopy(dict_common)
#                     for key in plot_vals.keys():
#                         dict_common_copy[key] = plot_vals[key][i]
#                         result_data.append(dict_common_copy)

# result_df = pd.DataFrame(result_data)
# df_filter = result_df["dataset_alpha"] == 1
# df_filter = (
#     (result_df["algorithm"] == "fedavg")
#     & (result_df["rounds"] == 1)
#     & (result_df["algo_seed"] == 0)
# )
# print(result_df[df_filter]["noise_level"])
# # sns.lineplot(
# #     data=filtered_result_df,
# #     x="rounds",
# #     y="train_acc",
# #     hue="algorithm",
# # )
# # plt.show()
# # for algorithm in algorithms:
# #     algo_results = result_dict[algorithm][num_selected]
# #     summary_results = average_results(algo_results)
# #     if algorithm == "ucb":
# #         algotext = r"""Fed-Shap-UCB, $\beta = 100$"""
# #     elif algorithm == "poc":
# #         algotext = r"""Power-Of-Choice"""
# #     elif algorithm == "fedavg":
# #         algotext = r"""FedAvg"""
# #     elif algorithm == "fedprox":
# #         algotext = r"FedProx, $\mu = 1$"
# #     elif algorithm == "sfedavg":
# #         algotext = r"S-FedAvg"

# # plt.plot(summary_results.test_loss, label=algotext)
# # plt.ylabel("Test Loss")
# # plt.xlabel("Communication Rounds")
# # # plt.ylim(2, 3)
# # plt.legend()
# # # plt.show()  # comment this for .tex generation
# # # generate .tex

# import tikzplotlib

# tikzplotlib.save(f"plots/mnist-noisy-1.tex")
# import matplotlib as mpl

# plt.close()
# mpl.rcParams.update(mpl.rcParamsDefault)


# """

# """

# # a = []
# # err = []
# # print(a)
# # for subdir, dirs, files in os.walk(rootdir):
# #     for file in files:
# #         file_path = os.path.join(subdir, file)
# #         with open(file_path, "rb") as f:
# #             results = pickle.load(f)
# #             algorithm = results.config["algorithm"]
# #             dataset = results.config["dataset"]
# #             num_clients = results.config["num_clients"]
# #             select_fraction = results.config["select_fraction"]
# #             num_selected = int(select_fraction * num_clients)
# #             T = results.config["T"]
# #             if num_selected == 12 and T == 20:
# #                 if algorithm == "ucb":
# #                     x = np.mean(results.num_model_evaluations["gtg"])
# #                     cosine_distance_gtg = np.mean(
# #                         results.cosine_distance(
# #                             results.sv_rounds["gtg"], results.sv_rounds["true"]
# #                         )
# #                     )
# #                     print(f"distances = {cosine_distance_gtg}")
# #                     a.append(x)
# #                     err.append(cosine_distance_gtg)
# # print(a)
# # print(np.mean(a))
# # print(np.mean(err))

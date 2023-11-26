import numpy as np
from scipy import stats
import pandas as pd
import wandb

from math import isnan
import pickle
import os
import pprint
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from utils import dict_hash

import warnings
warnings.filterwarnings("ignore")

rc("mathtext", default="regular")

result_path_parent = "results/"
dataset = "mnist"
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

def checknanequality(x, y):
    if x == y:
        return True
    elif isinstance(x, float) and isnan(x) and isinstance(y, float) and isnan(y):
        return True
    else:
        return False
def generate_timing_plots(result_df_original, smoothing=0, T = [150, 250, 350]):
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
    smoothed_df = smoothen_df(1, result_df)

    T_results = T_main - 1
    resultsgroup = smoothed_df[smoothed_df["_step"]==T_results].groupby(["algorithm", "memory","mu"], dropna = False)
    accuracy_mean = resultsgroup.mean()["test_accuracy"]
    best_idxs = accuracy_mean.groupby("algorithm").idxmax()

    smoothed_df_new = pd.DataFrame()
    # Smooth the data for each unique combination of 'algorithm' and 'algo_seed' separately using EWMA

    for (algorithm, seed, memory, mu, dataset_alpha, noise, systems_heterogenity), group in smoothed_df.groupby(
        ["algorithm", "seed", "memory","mu","dataset_alpha", "noise", "systems_heterogenity"], dropna = False
    ):
        algorithm_df = group
        # Sort the unique step values for interpolation
        sorted_steps = np.sort(algorithm_df["_step"].unique())

        # Perform EWMA smoothing on the accuracy
        smoothed_accuracy = algorithm_df["test_accuracy"].ewm(alpha=1-smoothing).mean()

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

        if (checknanequality(memory, best_idxs[algorithm][1])) and checknanequality(mu,best_idxs[algorithm][2]):
            smoothed_df_new = pd.concat([smoothed_df_new, algorithm_smoothed_df])

    g = sns.lineplot(
        data=smoothed_df_new, x="_step", y="test_accuracy", hue="algorithm", palette="tab10", errorbar="sd"
    )
    sns.despine()
    plt.ylabel("Test Accuracy")
    plt.xlabel("Communication Rounds")
    # plt.ylim([0, 1])
    fedprox_text = "FedProx"
    fedavg_text = "FedAvg"
    ucb_text = "UCB"
    poc_text = "Power-Of-Choice"
    sfedavg_text = "S-FedAvg"
    centralised_text = "Centralized"
    greedyshap_text = "GreedyFed"
    g.legend_.set_title("Algorithm")
    # ensure labels are in correct order
    new_labels = [
        centralised_text,
        fedavg_text,
        fedprox_text,
        greedyshap_text,
        poc_text,
        sfedavg_text,
        ucb_text,
    ]
    for t, l in zip(g.legend_.texts, new_labels):
        t.set_text(l)

    # plt.grid(True)
    plt.savefig(f'plots/{dataset}-timing.pdf', format="pdf", bbox_inches="tight")
    # plt.savefig('testplot.png')



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

# generate LaTeX tables from raw run data
print("data")
generate_datahet_results(result_df_original)
print("timing")
generate_timing_results(result_df_original, T=T_array)
print("systems")
generate_systems_results(result_df_original)
print("noise")
generate_noise_results(result_df_original)
print("plots")
generate_timing_plots(result_df_original, T=T_array)
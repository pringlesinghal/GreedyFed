import wandb

import torch
import os
import pandas as pd
import numpy as np
from copy import deepcopy


from initialise import initNetworkData
from algorithms import (
    fed_avg_run,
    fed_prox_run,
    sfedavg_run,
    ucb_run,
    greedy_shap_run,
    power_of_choice_run,
    centralised_run,
)
from utils import dict_hash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def execute_run(clients, server, algorithm, parameters, config_global, seed, wandb_logging=True):

    E = config_global["E"]
    B = config_global["batches"]
    T = config_global["T"]
    lr = config_global["lr"]
    momentum = config_global["momentum"]
    select_fraction = config_global["select_fraction"]

    test_acc_arr = []
    train_acc_arr = []
    train_loss_arr = []
    val_loss_arr = []
    test_loss_arr = []
    shap_arr = []
    selections_arr = []

    if algorithm == "centralised":
        if wandb_logging:
            wandb_config = deepcopy(config_global)
            wandb_config["algorithm"] = algorithm
            wandb_config["seed"] = seed
            wandb.init(project=GLOBAL_PROJECT_NAME, config=wandb_config)
        (
            test_acc,
            train_acc,
            train_loss,
            val_loss,
            test_loss,
            selections,
        ) = centralised_run(
            clients,
            server,
            select_fraction,
            T,
            random_seed=seed,
            E=E,
            B=B,
            learning_rate=lr,
            momentum=momentum,
            logging=wandb_logging,
        )
        test_acc_arr.append(test_acc)
        train_acc_arr.append(train_acc)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        test_loss_arr.append(test_loss)
        selections_arr.append(selections)

    elif algorithm == "fedavg":
        if wandb_logging:
            wandb_config = deepcopy(config_global)
            wandb_config["algorithm"] = algorithm
            wandb_config["seed"] = seed
            wandb.init(project=GLOBAL_PROJECT_NAME, config=wandb_config)
        (
            test_acc,
            train_acc,
            train_loss,
            val_loss,
            test_loss,
            selections,
        ) = fed_avg_run(
            clients,
            server,
            select_fraction,
            T,
            random_seed=seed,
            E=E,
            B=B,
            learning_rate=lr,
            momentum=momentum,
            logging=wandb_logging,
        )
        test_acc_arr.append(test_acc)
        train_acc_arr.append(train_acc)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        test_loss_arr.append(test_loss)
        selections_arr.append(selections)
    elif algorithm == "fedprox":
        mu_vals = parameters["mu"]
        for mu in mu_vals:
            if wandb_logging:
                wandb_config = deepcopy(config_global)
                wandb_config["algorithm"] = algorithm
                wandb_config["seed"] = seed
                wandb_config["mu"] = mu
                wandb.init(project=GLOBAL_PROJECT_NAME, config=wandb_config)
            (
                test_acc,
                train_acc,
                train_loss,
                val_loss,
                test_loss,
                selections,
            ) = fed_prox_run(
                clients,
                server,
                select_fraction,
                T,
                mu,
                random_seed=seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=wandb_logging,
            )
            test_acc_arr.append(test_acc)
            train_acc_arr.append(train_acc)
            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
            test_loss_arr.append(test_loss)
            selections_arr.append(selections)
    elif algorithm == "sfedavg":
        alpha_vals = parameters["alpha"]
        for alpha in alpha_vals:
            beta = 1 - alpha
            # temperature = 1
            # alpha_init = 1/clients.length
            if wandb_logging:
                wandb_config = deepcopy(config_global)
                wandb_config["algorithm"] = algorithm
                wandb_config["seed"] = seed
                wandb_config["algo_alpha"] = alpha
                wandb_config["algo_beta"] = beta
                wandb.init(project=GLOBAL_PROJECT_NAME, config=wandb_config)
            (
                test_acc,
                train_acc,
                train_loss,
                val_loss,
                test_loss,
                selections,
                shapley_values,
            ) = sfedavg_run(
                clients,
                server,
                select_fraction,
                T,
                alpha,
                beta,
                random_seed=seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=wandb_logging,
                # temperature=temperature,
                # alpha_init=alpha_init,
            )
            test_acc_arr.append(test_acc)
            train_acc_arr.append(train_acc)
            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
            test_loss_arr.append(test_loss)
            selections_arr.append(selections)
            shap_arr.append(shapley_values)
    elif algorithm == "ucb":
        beta_vals = parameters["beta"]
        for beta in beta_vals:
            if wandb_logging:
                wandb_config = deepcopy(config_global)
                wandb_config["algorithm"] = algorithm
                wandb_config["seed"] = seed
                wandb_config["algo_beta"] = beta
                wandb.init(project=GLOBAL_PROJECT_NAME, config=wandb_config)
            (
                test_acc,
                train_acc,
                train_loss,
                val_loss,
                test_loss,
                selections,
                shapley_values,
                sv_rounds,
                num_model_evaluations,
                ucb_values,
            ) = ucb_run(
                clients,
                server,
                select_fraction,
                T,
                beta,
                random_seed=seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=wandb_logging,
            )
            test_acc_arr.append(test_acc)
            train_acc_arr.append(train_acc)
            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
            test_loss_arr.append(test_loss)
            selections_arr.append(selections)
            shap_arr.append(shapley_values)
    elif algorithm == "greedyshap":
        shap_memory_vals = parameters["memory"]
        for shap_memory in shap_memory_vals:
            if wandb_logging:
                wandb_config = deepcopy(config_global)
                wandb_config["algorithm"] = algorithm
                wandb_config["seed"] = seed
                wandb_config["memory"] = shap_memory
                wandb.init(project=GLOBAL_PROJECT_NAME, config=wandb_config)
            (
                test_acc,
                train_acc,
                train_loss,
                val_loss,
                test_loss,
                selections,
                shapley_values,
                sv_rounds,
                num_model_evaluations,
                ucb_values,
            ) = greedy_shap_run(
                clients,
                server,
                select_fraction,
                T,
                random_seed=seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=wandb_logging,
                shap_memory=shap_memory,
            )
            test_acc_arr.append(test_acc)
            train_acc_arr.append(train_acc)
            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
            test_loss_arr.append(test_loss)
            selections_arr.append(selections)
            shap_arr.append(shapley_values)
    elif algorithm == "poc":
        decay_factor_vals = parameters["decay_factor"]
        for decay_factor in decay_factor_vals:
            if wandb_logging:
                wandb_config = deepcopy(config_global)
                wandb_config["algorithm"] = algorithm
                wandb_config["seed"] = seed
                wandb_config["decay_factor"] = decay_factor
                wandb.init(project=GLOBAL_PROJECT_NAME, config=wandb_config)
            (
                test_acc,
                train_acc,
                train_loss,
                val_loss,
                test_loss,
                selections,
            ) = power_of_choice_run(
                clients,
                server,
                select_fraction,
                T,
                decay_factor=decay_factor,
                random_seed=seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=wandb_logging,
            )
            test_acc_arr.append(test_acc)
            train_acc_arr.append(train_acc)
            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
            test_loss_arr.append(test_loss)
            selections_arr.append(selections)
    if algorithm in ["sfedavg", "ucb", "greedyshap"]:
        return test_acc_arr, train_acc_arr, train_loss_arr, val_loss_arr, test_loss_arr, selections_arr, shap_arr
    return test_acc_arr, train_acc_arr, train_loss_arr, val_loss_arr, test_loss_arr, selections_arr

# def save_to_excel(path, metrics, global_config, algorithm, parameters):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     df = pd.DataFrame(data=metrics)
#     df.to_excel(excel_writer=path+'test.xlsx', sheet_name='sheet1')


if __name__ == "__main__":
    wandb.login()
    wandb.finish()
    dataset = "mnist"
    systems_heterogenity_list = [0, 0, 0, 0.5, 0.9, 0, 0]
    dataset_alpha_list = [1e-4, 1e-1, 1e1, 1e-4, 1e-4, 1e-4, 1e-4]
    noise_list = [0,0,0,0,0, 0.1, 0.3]
    if dataset in ["fmnist","mnist"]:
        select_fraction = 0.01 # 0.03
        num_clients = 300
        T = 400
        if dataset in ["fmnist"]:
            GLOBAL_PROJECT_NAME = "FL-FMNIST-Final"
        else:
            GLOBAL_PROJECT_NAME = "FL-MNIST-Final"
    elif dataset in ["cifar10"]:
        select_fraction = 0.1
        num_clients = 200
        T = 200      
        GLOBAL_PROJECT_NAME = "FL-CIFAR10-Final"
    elif dataset in ["synthetic"]:        
        select_fraction = 0.03
        num_clients = 200
        T = 200 
        dataset_alpha_list = [1, 0.5, 0, 1, 1, 1, 1, 1] # only for synthetic
        GLOBAL_PROJECT_NAME = "FL-Synthetic-Final"

    for systems_heterogenity_item, dataset_alpha_item, noise_item in zip(systems_heterogenity_list, dataset_alpha_list, noise_list):
        dataset_config = {
            "dataset": dataset, # dataset from ["cifar10", "mnist", "synthetic", "fmnist"]
            "num_clients": num_clients,
            "dataset_alpha": dataset_alpha_item,
            "dataset_beta": dataset_alpha_item, # active only for synthetic
            "noise": noise_item,
            "systems_heterogenity": systems_heterogenity_item
        }

        algo_config_global = {
            "select_fraction":select_fraction,
            "epochs":5,
            "batches":5,
            "T":T,
            "lr": 0.01,
            "momentum":0.5,
        }

        algo_config_specific = {
            "fedavg":{},
            "fedprox":{"mu":[1e-2, 1e-1, 1, 10]},
            "sfedavg":{"alpha":[0.5]},
            "poc":{"decay_factor":[0.9]},
            "ucb":{"beta":[1e-2, 1]},
            "greedyshap":{"memory":["mean",0, 0.1, 0.5, 0.9]},
            "centralised":{}
        }
        

        num_seeds = 5
        # path = './local_logs/'
        systems_heterogenity = dataset_config["systems_heterogenity"]
        global_config = {**dataset_config, **algo_config_global}
        for seed in range(num_seeds):
            data_seed = seed
            if systems_heterogenity > 0:
                np.random.seed(seed)
                num_clients = dataset_config["num_clients"]
                num_slow = int(np.floor(systems_heterogenity * num_clients))
                slow_clients = np.random.choice(a=num_clients,size=num_slow,replace=False)
                E_base = deepcopy(algo_config_global["epochs"])
                E = [E_base for i in range(num_clients)]
                for i in slow_clients:
                    E[i] = np.random.choice(a=range(1,E_base+1), size=1)[0]
                global_config["E"] = E
            else:
                global_config["E"] = global_config["epochs"]
            clients, server = initNetworkData( 
                    dataset=dataset_config["dataset"],
                    num_clients=dataset_config["num_clients"],
                    random_seed=data_seed,
                    alpha=dataset_config["dataset_alpha"],
                    beta=dataset_config["dataset_beta"],
                    update_noise_level=dataset_config["noise"],
                )
            for algorithm, parameters in algo_config_specific.items():
                clients_copy = deepcopy(clients)
                server_copy = deepcopy(server)
                metrics = execute_run(clients=clients_copy, server=server_copy, algorithm=algorithm, parameters=parameters, config_global=global_config, seed=seed)
                # save_to_excel(path=path, metrics=metrics, global_config=global_config, algorithm=algorithm, parameters=parameters)

        wandb.init(project="FL-RUN-COMPLETED", name=f"finishing-{dataset_config['dataset']}")
        wandb.alert(title=f"finishing-{dataset_config['dataset']}", text="Finishing")
        wandb.finish()

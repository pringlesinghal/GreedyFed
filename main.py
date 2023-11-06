import wandb

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os

GLOBAL_PROJECT_NAME = "FL-FMNIST-Nov"
from initialise import initNetworkData
from algorithms import (
    fed_avg_run,
    fed_prox_run,
    sfedavg_run,
    ucb_run,
    power_of_choice_run,
    centralised_run,
)
from utils import dict_hash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlgoResults:
    def __init__(self, test_acc, train_acc, train_loss, val_loss, test_loss):
        self.test_acc = test_acc
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss

    def get_results(self):
        return (
            self.test_acc,
            self.train_acc,
            self.train_loss,
            self.val_loss,
            self.test_loss,
        )

    def plot_accuracy(self):
        test_acc = self.test_acc
        train_acc = self.train_acc
        plt.plot(test_acc, label="test accuracy")
        plt.plot(train_acc, label="train accuracy")
        plt.legend()
        plt.show()

    def plot_loss(self):
        train_loss = self.train_loss
        val_loss = self.val_loss
        test_loss = self.test_loss
        plt.plot(train_loss, label="train loss")
        plt.plot(val_loss, label="val loss")
        plt.plot(test_loss, label="test loss")
        plt.legend()
        plt.show()

    def compute_sv_metrics(self):
        if self.config["algorithm"] == "ucb":
            cosine_distances_gtg = self.cosine_distance(
                self.sv_rounds["gtg"], self.sv_rounds["true"]
            )
            cosine_distances_tmc = self.cosine_distance(
                self.sv_rounds["tmc"], self.sv_rounds["true"]
            )
            self.cosine_distance_gtg = np.mean(cosine_distances_gtg)
            self.cosine_distance_tmc = np.mean(cosine_distances_tmc)
            self.num_evals_gtg = np.mean(np.array(self.num_model_evaluations["gtg"]))
            self.num_evals_tmc = np.mean(np.array(self.num_model_evaluations["tmc"]))
            self.num_evals_true = np.mean(np.array(self.num_model_evaluations["true"]))

            plt.plot(cosine_distances_gtg, label="cosine distance gtg")
            plt.plot(cosine_distances_tmc, label="cosine distance tmc")
            plt.legend()
            plt.show()

            for sv_method in ["gtg", "true", "tmc"]:
                plt.plot(
                    self.num_model_evaluations[sv_method],
                    label=f"{sv_method} numb model evals",
                )
            plt.legend()
            plt.show()

    def cosine_distance(self, sv_1, sv_2):
        num_sequences = len(sv_1)
        assert num_sequences == len(sv_2)
        cosine_distances = []
        for i in range(num_sequences):
            sv_1_norm = np.linalg.norm(np.array(sv_1[i]))
            sv_2_norm = np.linalg.norm(np.array(sv_2[i]))
            distance = 1 - np.dot(sv_1[i], sv_2[i]) / (sv_1_norm * sv_2_norm)
            cosine_distances.append(distance)
        return cosine_distances


class AlgoRun:
    def __init__(
        self,
        dataset_config,
        algorithm,
        select_fraction,
        algo_seed=0,
        data_seed=0,
        E=10,
        B=10,
        T=100,
        lr=0.01,
        momentum=0.5,
        mu=None,
        alpha=None,
        beta=None,
        decay_factor=None,
        noise_level=None,
        temperature=None,
        alpha_init=None,
        shap_memory=None,
    ):
        """
        dataset_config = dict with keys {dataset, num_clients, alpha, beta}
        """
        self.dataset_config = dataset_config
        self.data_seed = data_seed
        self.algorithm = algorithm
        self.select_fraction = select_fraction
        self.algo_seed = algo_seed
        # additional parameters
        self.E = E
        self.B = B
        self.T = T
        self.lr = lr
        self.momentum = momentum
        # algorithm parameters
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.decay_factor = decay_factor
        self.temperature = temperature
        self.alpha_init = alpha_init
        self.shap_memory = shap_memory
        # noise parameters
        self.noise_level = noise_level
        if self.algorithm in ["fedavg", "centralised"]:
            self.mu = None
            self.alpha = None
            self.beta = None
            self.decay_factor = None
        elif self.algorithm == "fedprox":
            if mu is None:
                raise Exception("FedProx requires mu to be passed")
            self.mu = mu
            self.alpha = None
            self.beta = None
            self.decay_factor = None
        elif self.algorithm == "sfedavg":
            if (alpha is None) and (beta is None):
                raise Exception(
                    "S-FedAvg requires either alpha or beta to be specified"
                )
            elif alpha is None:
                self.beta = beta
                self.alpha = 1 - self.beta
            elif beta is None:
                self.alpha = alpha
                self.beta = 1 - self.alpha
            else:
                self.alpha = alpha
                self.beta = beta
                self.mu = None
                self.decay_factor = None
        elif algorithm == "poc":
            if decay_factor is None:
                raise Exception("poc requires decay_factor to be specified")
            self.decay_factor = decay_factor
            self.mu = None
            self.alpha = None
            self.beta = None
        elif algorithm == "ucb":
            if self.beta is None:
                raise Exception("ucb requires beta to be specified")
            self.beta = beta
            self.alpha = None
            self.mu = None
            self.decay_factor = None
        else:
            raise Exception("Unknown algorithm")

    def run(self, logging):
        """
        logging must be one of the following:
            1. False: no logging on wandb
            2. True: log the run on wandb
        """
        dataset_config = self.dataset_config
        data_seed = self.data_seed
        clients, server = initNetworkData(
            dataset=dataset_config["dataset"],
            num_clients=dataset_config["num_clients"],
            random_seed=data_seed,
            alpha=dataset_config["alpha"],
            beta=dataset_config["beta"],
            update_noise_level=self.noise_level,
        )
        algorithm = self.algorithm
        random_seed = self.algo_seed

        E = self.E
        B = self.B
        select_fraction = self.select_fraction
        T = self.T
        lr = self.lr
        momentum = self.momentum
        wandb_config = {
            "algorithm": self.algorithm,
            "dataset": self.dataset_config["dataset"],
            "num_clients": self.dataset_config["num_clients"],
            "dataset_alpha": self.dataset_config["alpha"],
            "dataset_beta": self.dataset_config["beta"],
            "algo_seed": self.algo_seed,
            "data_seed": self.data_seed,
            "E": E,
            "B": B,
            "select_fraction": select_fraction,
            "T": T,
            "lr": lr,
            "momentum": momentum,
            "mu": self.mu,
            "algo_alpha": self.alpha,
            "algo_beta": self.beta,
            "decay_factor": self.decay_factor,
            "temperature": self.temperature,
            "alpha_init": self.alpha_init,
            "shap_memory": self.shap_memory,
        }

        if self.noise_level is not None:
            wandb_config["noise_level"] = self.noise_level
        else:
            wandb_config["noise_level"] = 0

        # result_path = f'results/{self.dataset_config["dataset"]}/{self.algorithm}/{self.dataset_config["num_clients"]}-{int(self.select_fraction*self.dataset_config["num_clients"])}/'
        # result_path = f'results-cifar10-final/{self.algorithm}/select-{int(self.select_fraction*self.dataset_config["num_clients"])}/'
        # if os.path.exists(result_path + f"{dict_hash(wandb_config)}.pickle"):
        #     print("this run has been performed earlier")
        #     with open(result_path + f"{dict_hash(wandb_config)}.pickle", "rb") as f:
        #         self.results = pickle.load(f)

        #     return self.results.get_results()

        if logging:
            wandb.init(project=GLOBAL_PROJECT_NAME, config=wandb_config)

        if algorithm == "centralised":
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
                random_seed=random_seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=logging,
            )

        elif algorithm == "fedavg":
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
                random_seed=random_seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=logging,
            )
        elif algorithm == "fedprox":
            mu = self.mu
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
                random_seed=random_seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=logging,
            )
        elif algorithm == "sfedavg":
            alpha = self.alpha
            beta = self.beta
            temperature = self.temperature
            alpha_init = self.alpha_init
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
                random_seed=random_seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=logging,
                temperature=temperature,
                alpha_init=alpha_init,
            )
        elif algorithm == "ucb":
            beta = self.beta
            shap_memory = self.shap_memory
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
                random_seed=random_seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=logging,
                shap_memory=shap_memory,
            )
        elif algorithm == "poc":
            decay_factor = self.decay_factor
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
                random_seed=random_seed,
                E=E,
                B=B,
                learning_rate=lr,
                momentum=momentum,
                logging=logging,
            )

        self.results = AlgoResults(test_acc, train_acc, train_loss, val_loss, test_loss)

        # log the selections for each algorithm
        self.results.selections = selections

        # for sfedavg and ucb
        #   log the SV
        #   also log the number of model evaluations for gtg, tmc, true as well as cosine distance between gtg|true, tmc|true
        if algorithm in ["sfedavg", "ucb"]:
            self.results.shapley_values = shapley_values
        if algorithm == "ucb":
            self.results.sv_rounds = sv_rounds
            self.results.num_model_evaluations = num_model_evaluations
            self.results.ucb_values = ucb_values
        # if logging == True:
        #     self.results.config = wandb_config
        #     # result_path = f'results/{self.dataset_config["dataset"]}/{self.algorithm}/{self.dataset_config["num_clients"]}-{int(self.select_fraction*self.dataset_config["num_clients"])}/'
        #     result_path = f'results-cifar10-final/{self.algorithm}/select-{int(self.select_fraction*self.dataset_config["num_clients"])}/'
        #     os.makedirs(result_path, exist_ok=True)
        #     with open(result_path + f"{dict_hash(wandb_config)}.pickle", "wb") as f:
        #         pickle.dump(self.results, f)
        #     # save results to local file
        return test_acc, train_acc, train_loss, val_loss, test_loss


def avg_runs(num_runs, algorun, logging):
    """
    Takes AlgoRun template algorun and performs multiple runs with different data and algorithm seeds
    """
    test_acc_avg = []
    train_acc_avg = []
    train_loss_avg = []
    val_loss_avg = []
    test_loss_avg = []

    for run in range(num_runs):
        algorun.data_seed = run
        algorun.algo_seed = run
        test_acc, train_acc, train_loss, val_loss, test_loss = algorun.run(logging)
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
    return test_acc_avg, train_acc_avg, train_loss_avg, val_loss_avg, test_loss_avg


if __name__ == "__main__":
    wandb.login()
    wandb.finish()
    """Experiments"""

    """
    Experiment 1
    """
    """
    First configure dataset and split
    """
    # dataset from ["cifar10", "mnist", "synthetic", "fmnist"]
    dataset = "fmnist"
    num_clients = 300
    dirichlet_alpha = 0.0001
    dataset_alpha = 1
    dataset_beta = 1  # needed for synthetic dataset
    if dataset != "synthetic":
        dataset_alpha = dirichlet_alpha

    """
    Then configure the algorithm
    """
    algorithm = "fedavg"
    select_fraction = 10 / 360

    E = 5
    B = 5
    T = 500
    lr = 0.01
    momentum = 0.5
    mu = None
    alpha = None
    beta = None
    decay_factor = None

    noise_level = 1e-1

    """
    Perform runs
    """
    num_runs = 5

    noise_levels = [0, 0.1]
    dataset_alphas = [1e-4, 1e-1, 1e2]  # 1e-4,
    algorithms = [
        "ucb",
        # "centralised",
        # "sfedavg",
        # "fedavg",
    ]  # ["ucb" , "fedavg", "centralised", "poc", "fedprox", "sfedavg"]
    select_fractions = np.array([4, 1, 2, 7, 20]) / num_clients  # 4, 7, 20
    sfedavg_alphas = [0.5]  # [0.1, 0.5, 0.9]
    ### these two are zipped together
    temperatures = [1]
    alpha_inits = [1 / num_clients]
    ###
    poc_decay_factors = [0.9]
    fedprox_mus = [10]
    ucb_beta = 0  # [1e-6, 0.001, 0.01, 10, 100]
    ucb_shap_memory = ["mean-norm"]  # ["mean", 0, 0.4, 0.8]

    for select_fraction in select_fractions:
        for noise_level in noise_levels:
            for dataset_alpha in dataset_alphas:
                dataset_config = {
                    "dataset": dataset,
                    "num_clients": num_clients,
                    "alpha": dataset_alpha,
                    "beta": dataset_alpha,  # keeping alpha = beta
                }
                for algorithm in algorithms:
                    if algorithm == "sfedavg":
                        for alpha in sfedavg_alphas:
                            for temperature, alpha_init in zip(
                                temperatures, alpha_inits
                            ):
                                beta = 1 - alpha

                                test_run = AlgoRun(
                                    dataset_config,
                                    algorithm,
                                    select_fraction,
                                    E=E,
                                    B=B,
                                    T=T,
                                    lr=lr,
                                    momentum=momentum,
                                    mu=mu,
                                    alpha=alpha,
                                    beta=beta,
                                    decay_factor=decay_factor,
                                    noise_level=noise_level,
                                    temperature=temperature,
                                    alpha_init=alpha_init,
                                )
                                avg_runs(num_runs, test_run, logging=True)

                    elif algorithm in ["fedavg", "centralised"]:
                        test_run = AlgoRun(
                            dataset_config,
                            algorithm,
                            select_fraction,
                            E=E,
                            B=B,
                            T=T,
                            lr=lr,
                            momentum=momentum,
                            mu=mu,
                            alpha=alpha,
                            beta=beta,
                            decay_factor=decay_factor,
                            noise_level=noise_level,
                        )
                        avg_runs(num_runs, test_run, logging=True)
                    elif algorithm == "poc":
                        for decay_factor in poc_decay_factors:
                            test_run = AlgoRun(
                                dataset_config,
                                algorithm,
                                select_fraction,
                                E=E,
                                B=B,
                                T=T,
                                lr=lr,
                                momentum=momentum,
                                mu=mu,
                                alpha=alpha,
                                beta=beta,
                                decay_factor=decay_factor,
                                noise_level=noise_level,
                            )
                            avg_runs(num_runs, test_run, logging=True)
                    elif algorithm == "fedprox":
                        for mu in fedprox_mus:
                            test_run = AlgoRun(
                                dataset_config,
                                algorithm,
                                select_fraction,
                                E=E,
                                B=B,
                                T=T,
                                lr=lr,
                                momentum=momentum,
                                mu=mu,
                                alpha=alpha,
                                beta=beta,
                                decay_factor=decay_factor,
                                noise_level=noise_level,
                            )
                            avg_runs(num_runs, test_run, logging=True)
                    elif algorithm == "ucb":
                        beta = ucb_beta
                        for shap_memory in ucb_shap_memory:
                            test_run = AlgoRun(
                                dataset_config,
                                algorithm,
                                select_fraction,
                                E=E,
                                B=B,
                                T=T,
                                lr=lr,
                                momentum=momentum,
                                mu=mu,
                                alpha=alpha,
                                beta=beta,
                                decay_factor=decay_factor,
                                noise_level=noise_level,
                                shap_memory=shap_memory,
                            )
                            avg_runs(num_runs, test_run, logging=True)

    wandb.init(project="FL-RUN-COMPLETED", name="finishing-syn-full-search")
    wandb.alert(title="finishing synthetic full search", text="Finishing syn runs")
    wandb.finish()

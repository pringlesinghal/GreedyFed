import pickle
import matplotlib.pyplot as plt
import numpy as np

methods = ["ucb", "sfedavg", "fedavg", "poc", "fedprox"]
dataset = "synthetic"
num_clients = 40
random_seed = 1
dirichlet_alpha = 1
synthetic_alpha = 1
synthetic_beta = 1
accuracies = {}
for method in methods:
    if dataset in ["mnist", "cifar10"]:
        file_path = f"./results/{method}_{dataset}_{num_clients}_{random_seed}_{dirichlet_alpha}.pickle"
    else:
        file_path = f"./results/{method}_{dataset}_{num_clients}_{random_seed}_{synthetic_alpha}_{synthetic_beta}.pickle"
    with open(file_path, "rb") as f:
        accuracies[method] = pickle.load(f)
# # best hyperparams for cifar10_40_1_1
# ucb_beta = 10
# sfedavg_alpha = 0.9
# sfedavg_beta = 0.1
# poc_lambda = 0.95
# fedprox_mu = 0.001

# # best hyperparams for synthetic_40_0.5_0.5
# ucb_beta = 100
# sfedavg_alpha = 0.5
# sfedavg_beta = 0.5
# poc_lambda = 0.99
# fedprox_mu = 1

# best hyperparams for synthetic_40_1_1
ucb_beta = 1
sfedavg_alpha = 0.9
sfedavg_beta = 0.1
poc_lambda = 0.8
fedprox_mu = 1


# fedprox plot
acc_fedprox = accuracies["fedprox"]
for key in [fedprox_mu]:
    mu = key
    plt.plot(acc_fedprox[key], label="FedProx: " + r"$\mu$ = " + f"{mu}")

# poc plot
acc_poc = accuracies["poc"]
for key in [poc_lambda]:
    decay_rate = key
    plt.plot(
        acc_poc[key], label="Power-of-Choice: " + r"$\lambda$ = " + f"{decay_rate}"
    )

# fedavg plot
plt.plot(accuracies["fedavg"], label="FedAvg")

# sfedavg plot
acc_sfedavg = accuracies["sfedavg"]
for key in acc_sfedavg.keys():
    alpha, beta = key
    if np.abs(alpha - sfedavg_alpha) < 1e-5:
        plt.plot(
            acc_sfedavg[key],
            label="S-FedAvg: " + r"$\alpha$, $\beta = $" + f"{alpha:.1f},{beta:.1f}",
        )


# ucb plot
acc_ucb = accuracies["ucb"]
for beta in [ucb_beta]:
    plt.plot(acc_ucb[beta], label="UCB: " + r"$\beta$ = " + f"{beta}")

plt.legend()
plt.xlabel("Communication Rounds")
plt.ylabel("Test Accuracy")
plt.show()

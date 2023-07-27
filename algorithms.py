import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy

from utils import topk


def fed_prox_criterion(model_reference, mu):
    """
    returns the required function when called
    loss function for FedProx with chosen mu parameter
    """
    model_reference = deepcopy(model_reference)

    def loss(model, data, targets):
        criterion = torch.nn.CrossEntropyLoss()
        scores = model(data)
        loss_value = criterion(scores, targets)
        for param, param_reference in zip(
            model.parameters(), model_reference.parameters()
        ):
            loss_value += (
                0.5 * mu * torch.square(torch.linalg.norm((param - param_reference)))
            )
        return loss_value

    return loss


def fed_avg_criterion():
    def loss(model, data, targets):
        criterion = torch.nn.CrossEntropyLoss()
        scores = model(data)
        return criterion(scores, targets)

    return loss


def centralised_run(
    clients,
    server,
    select_fraction,
    T,
    random_seed=0,
    E=5,
    B=10,
    learning_rate=0.01,
    momentum=0.5,
    logging=False,
):
    clients = deepcopy(clients)
    server = deepcopy(server)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    data = []
    targets = []
    for client in clients:
        data.append(client.data)
        targets.append(client.targets)
    data = torch.cat(data)
    targets = torch.cat(targets)
    num_datapoints = len(data)
    print(num_datapoints)
    num_selected = int(np.floor(select_fraction * num_datapoints))
    print(num_selected)

    test_acc = []
    train_acc = []
    train_loss = []
    val_loss = []
    test_loss = []

    selections = []
    optimiser = optim.SGD(
        server.model.parameters(), lr=learning_rate, momentum=momentum
    )
    for t in tqdm(range(T)):
        for iteration in range(E * B):
            all_indices = np.random.permutation(list(range(num_datapoints)))
            indices = all_indices[:num_selected]
            data_raw = [data[j] for j in indices]
            targets_raw = [int(targets[j]) for j in indices]
            # prepare data and targets for training
            data_batch = (
                torch.stack(data_raw, 0).to(device=server.device).to(torch.float32)
            )
            targets_batch = torch.tensor(targets_raw).to(device=server.device)
            optimiser.zero_grad()
            loss = fed_avg_criterion()(server.model, data_batch, targets_batch)
            loss.backward()
            optimiser.step()

        test_acc_now = server.accuracy()
        train_acc_now = server.accuracy()
        with torch.no_grad():
            train_loss_now = float(
                fed_avg_criterion()(server.model, data, targets).cpu()
            )
        val_loss_now = server.val_loss(server.model, fed_avg_criterion())
        test_loss_now = server.test_loss(fed_avg_criterion())

        train_acc.append(train_acc_now)
        test_acc.append(test_acc_now)
        train_loss.append(train_loss_now)
        val_loss.append(val_loss_now)
        test_loss.append(test_loss_now)

        log_dict = {
            "train_accuracy": train_acc_now,
            "test_accuracy": test_acc_now,
            "train_loss": train_loss_now,
            "val_loss": val_loss_now,
            "test_loss": test_loss_now,
        }
        if logging == True:
            wandb.log(log_dict)

    if logging == True:
        print("finishing")
        wandb.finish()

    return test_acc, train_acc, train_loss, val_loss, test_loss, selections


def fed_avg_run(
    clients,
    server,
    select_fraction,
    T,
    random_seed=0,
    E=5,
    B=10,
    learning_rate=0.01,
    momentum=0.5,
    logging=False,
):
    clients = deepcopy(clients)
    client_weights = np.array([client.length for client in clients])
    client_weights = client_weights / np.sum(client_weights)
    server = deepcopy(server)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_clients = len(clients)
    num_selected = int(np.ceil(select_fraction * num_clients))

    test_acc = []
    train_acc = []
    train_loss = []
    val_loss = []
    test_loss = []

    selections = []
    for t in tqdm(range(T)):
        # select clients to transmit weights to

        # uniform random
        all_clients = [i for i in range(num_clients)]
        np.random.shuffle(all_clients)
        selected_client_indices = all_clients[0:num_selected]
        selected_status = [False for i in range(num_clients)]
        for i in range(num_clients):
            if i in selected_client_indices:
                selected_status[i] = True

        selections.append(np.array(selected_status).astype(int))

        client_states = []
        weights = []
        for idx, client in enumerate(clients):
            if selected_status[idx]:
                # perform descent at client
                client_state = client.train(
                    server.model,
                    criterion=fed_avg_criterion(),
                    E=E,
                    B=B,
                    learning_rate=learning_rate,
                    momentum=momentum,
                )
                weight = client.length  # number of data points at client
                client_states.append(client_state)
                weights.append(weight)

        server.aggregate(client_states, weights)

        test_acc_now = server.accuracy()
        train_acc_now = np.sum(
            [
                client_weights[i] * clients[i].accuracy_(server.model)
                for i in range(num_clients)
            ]
        )
        train_loss_now = np.sum(
            [
                client_weights[i] * clients[i].loss(server.model, fed_avg_criterion())
                for i in range(num_clients)
            ]
        )
        val_loss_now = server.val_loss(server.model, fed_avg_criterion())
        test_loss_now = server.test_loss(fed_avg_criterion())

        train_acc.append(train_acc_now)
        test_acc.append(test_acc_now)
        train_loss.append(train_loss_now)
        val_loss.append(val_loss_now)
        test_loss.append(test_loss_now)

        log_dict = {
            "train_accuracy": train_acc_now,
            "test_accuracy": test_acc_now,
            "train_loss": train_loss_now,
            "val_loss": val_loss_now,
            "test_loss": test_loss_now,
        }
        if logging == True:
            wandb.log(log_dict)

    if logging == True:
        print("finishing")
        wandb.finish()

    return test_acc, train_acc, train_loss, val_loss, test_loss, selections


def fed_prox_run(
    clients,
    server,
    select_fraction,
    T,
    mu,
    random_seed=0,
    E=5,
    B=10,
    learning_rate=0.01,
    momentum=0.5,
    logging=False,
):
    clients = deepcopy(clients)
    client_weights = np.array([client.length for client in clients])
    client_weights = client_weights / np.sum(client_weights)
    server = deepcopy(server)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_clients = len(clients)
    num_selected = int(np.ceil(select_fraction * num_clients))

    test_acc = []
    train_acc = []
    train_loss = []
    val_loss = []
    test_loss = []

    selections = []
    for t in tqdm(range(T)):
        # select clients to transmit weights to

        # uniform random
        all_clients = [i for i in range(num_clients)]
        np.random.shuffle(all_clients)
        selected_client_indices = all_clients[0:num_selected]
        selected_status = [False for i in range(num_clients)]
        for i in range(num_clients):
            if i in selected_client_indices:
                selected_status[i] = True

        selections.append(np.array(selected_status).astype(int))

        client_states = []
        weights = []
        for idx, client in enumerate(clients):
            if selected_status[idx]:
                # perform descent at client
                client_state = client.train(
                    server.model,
                    criterion=fed_prox_criterion(server.model, mu=mu),
                    E=E,
                    B=B,
                    learning_rate=learning_rate,
                    momentum=momentum,
                )
                weight = client.length  # number of data points at client
                client_states.append(client_state)
                weights.append(weight)

        server.aggregate(client_states, weights)
        test_acc_now = server.accuracy()
        train_acc_now = np.sum(
            [
                client_weights[i] * clients[i].accuracy_(server.model)
                for i in range(num_clients)
            ]
        )
        train_loss_now = np.sum(
            [
                client_weights[i] * clients[i].loss(server.model, fed_avg_criterion())
                for i in range(num_clients)
            ]
        )
        val_loss_now = server.val_loss(server.model, fed_avg_criterion())
        test_loss_now = server.test_loss(fed_avg_criterion())

        train_acc.append(train_acc_now)
        test_acc.append(test_acc_now)
        train_loss.append(train_loss_now)
        val_loss.append(val_loss_now)
        test_loss.append(test_loss_now)

        # # adpative mu
        # try:
        #     if val_loss[-1] > val_loss[-2]:
        #         mu += 0.1
        #     elif val_loss[-1] < val_loss[-5]:
        #         mu -= 0.1
        # except IndexError:
        #     pass
        # finally:
        #     print(f"mu = {mu}")

        log_dict = {
            "train_accuracy": train_acc_now,
            "test_accuracy": test_acc_now,
            "train_loss": train_loss_now,
            "val_loss": val_loss_now,
            "test_loss": test_loss_now,
        }
        if logging == True:
            wandb.log(log_dict)

    if logging == True:
        wandb.finish()

    return test_acc, train_acc, train_loss, val_loss, test_loss, selections


def power_of_choice_run(
    clients,
    server,
    select_fraction,
    T,
    decay_factor=1,
    random_seed=0,
    E=5,
    B=10,
    learning_rate=0.01,
    momentum=0.5,
    logging=False,
):
    """
    Power of Choice
    decay_factor (default = 1, no decay)
        determines the decay rate of number of clients to transmit the server model to (choose_from)
    """
    clients = deepcopy(clients)
    client_weights = np.array([client.length for client in clients])
    client_weights = client_weights / np.sum(client_weights)
    server = deepcopy(server)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_clients = len(clients)
    num_selected = int(np.ceil(select_fraction * num_clients))

    test_acc = []
    train_acc = []
    train_loss = []
    val_loss = []
    test_loss = []

    selections = []

    choose_from = num_clients  # the size of initial client subset to query for loss
    for t in tqdm(range(T)):
        # select clients to transmit weights to
        if choose_from > num_selected:
            choose_from *= decay_factor
            choose_from = np.max([int(np.ceil(choose_from)), num_selected])
        # uniform random
        all_clients = [i for i in range(num_clients)]
        np.random.shuffle(all_clients)
        selected_client_indices = all_clients[0:choose_from]
        selected_status = [False for i in range(num_clients)]
        for i in range(num_clients):
            if i in selected_client_indices:
                selected_status[i] = True

        selections.append(np.array(selected_status).astype(int))

        client_losses = []  # will store array of size choose_from
        for idx, client in enumerate(clients):
            if selected_status[idx]:
                # query selected clients for loss
                client_loss = client.loss(server.model, fed_avg_criterion())
                client_losses.append(client_loss)
        # find indices of largest num_selected values in client_losses
        indices = topk(client_losses, num_selected)
        selected_client_indices_2 = []  # will store array of size num_selected
        for i in indices:
            selected_client_indices_2.append(selected_client_indices[i])

        selected_status = [False for i in range(num_clients)]
        for i in range(num_clients):
            if i in selected_client_indices_2:
                selected_status[i] = True

        client_states = []
        weights = []
        for idx, client in enumerate(clients):
            if selected_status[idx]:
                # perform descent at client
                client_state = client.train(
                    server.model,
                    criterion=fed_avg_criterion(),
                    E=E,
                    B=B,
                    learning_rate=learning_rate,
                    momentum=momentum,
                )
                weight = client.length  # number of data points at client
                client_states.append(client_state)
                weights.append(weight)

        server.aggregate(client_states, weights)
        test_acc_now = server.accuracy()
        train_acc_now = np.sum(
            [
                client_weights[i] * clients[i].accuracy_(server.model)
                for i in range(num_clients)
            ]
        )
        train_loss_now = np.sum(
            [
                client_weights[i] * clients[i].loss(server.model, fed_avg_criterion())
                for i in range(num_clients)
            ]
        )
        val_loss_now = server.val_loss(server.model, fed_avg_criterion())
        test_loss_now = server.test_loss(fed_avg_criterion())

        train_acc.append(train_acc_now)
        test_acc.append(test_acc_now)
        train_loss.append(train_loss_now)
        val_loss.append(val_loss_now)
        test_loss.append(test_loss_now)

        log_dict = {
            "train_accuracy": train_acc_now,
            "test_accuracy": test_acc_now,
            "train_loss": train_loss_now,
            "val_loss": val_loss_now,
            "test_loss": test_loss_now,
        }
        if logging == True:
            wandb.log(log_dict)

    if logging == True:
        wandb.finish()

    return test_acc, train_acc, train_loss, val_loss, test_loss, selections


"""

def shapley_run(
    clients,
    server,
    select_fraction,
    T,
    client_selection,
    random_seed=0,
    E=5,
    B=10,
    learning_rate=0.01,
    momentum=0.5,
    logging=False,
):
    clients = deepcopy(clients)
    server = deepcopy(server)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_clients = len(clients)
    num_selected = int(np.ceil(select_fraction * num_clients))
    choose_from = num_clients  # the size of initial client subset to query for loss

    accuracy = []
    val_loss = []
    test_loss = []
    shapley_values_T = []
    selections_T = []
    for t in tqdm(range(T)):
        # select clients to transmit weights to
        # uniform random
        client_states = []
        weights = []
        client_losses = []
        for idx, client in enumerate(clients):
            client_losses.append(client.loss(server.model, fed_avg_criterion()))
            # perform descent at client
            client_state = client.train(
                server.model,
                criterion=fed_avg_criterion(),
                E=E,
                B=B,
                learning_rate=learning_rate,
                momentum=momentum,
            )
            weight = client.length  # number of data points at client
            client_states.append(client_state)
            weights.append(weight)
        # compute shapley values for each client
        # shapley_values = server.shapley_values_mc(
        #     fed_avg_criterion(), client_states, weights
        # )
        # shapley_values = server.shapley_values_tmc(
        #     fed_avg_criterion(), client_states, weights
        # )
        shapley_values = server.shapley_values_true(
            fed_avg_criterion(), client_states, weights
        )
        shapley_values_T.append(shapley_values)

        # find indices of largest num_selected values in shapley_values
        selections = [0 for i in range(num_clients)]
        if client_selection == "best":
            indices = topk(shapley_values, num_selected)
        elif client_selection == "fedavg":
            indices = np.random.choice(num_clients, size=num_selected, replace=False)
        elif client_selection == "worst":
            indices = np.argpartition(shapley_values, num_selected)[:num_selected]
        elif client_selection == "power_of_choice":
            indices = topk(client_losses, num_selected)
        client_states_chosen = [client_states[i] for i in indices]
        weights_chosen = [weights[i] for i in indices]

        for idx in indices:
            selections[idx] = 1
        selections_T.append(selections)

        server.aggregate(client_states_chosen, weights_chosen)
        accuracy_now = server.accuracy()
        val_loss_now = server.val_loss(server.model, fed_avg_criterion())
        test_loss_now = server.test_loss(fed_avg_criterion())
        accuracy.append(accuracy_now)
        val_loss.append(val_loss_now)
        test_loss.append(test_loss_now)

        log_dict = {
            "accuracy": accuracy_now,
            "val_loss": val_loss_now,
            "test_loss": test_loss_now,
        }
        for i in range(num_clients):
            log_dict[f"shapley_value_{i}"] = shapley_values[i]
            log_dict[f"selection_{i}"] = selections[i]
        if logging == True:
            wandb.log(log_dict)

    if logging == True:
        wandb.finish()
    return accuracy, val_loss, test_loss, shapley_values_T, selections_T

"""


def ucb_run(
    clients,
    server,
    select_fraction,
    T,
    beta,
    random_seed=0,
    E=5,
    B=10,
    learning_rate=0.01,
    momentum=0.5,
    logging=False,
):
    clients = deepcopy(clients)
    client_weights = np.array([client.length for client in clients])
    client_weights = client_weights / np.sum(client_weights)
    server = deepcopy(server)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_clients = len(clients)
    num_selected = int(np.ceil(select_fraction * num_clients))

    test_acc = []
    train_acc = []
    train_loss = []
    val_loss = []
    test_loss = []

    shapley_values_T = []
    ucb_values_T = []
    selections_T = []
    draws_T = []
    sv_rounds = {"gtg": [], "tmc": [], "true": []}
    num_model_evaluations = {"gtg": [], "tmc": [], "true": []}

    N_t = [0 for i in range(num_clients)]
    UCB = [0 for i in range(num_clients)]
    SV = [0 for i in range(num_clients)]
    for t in tqdm(range(T)):
        # select clients to transmit weights to
        # initially sample every client atleast once
        selected_status = [False for i in range(num_clients)]
        if t < np.floor(num_clients / num_selected):
            for idx in range(t * num_selected, (t + 1) * num_selected):
                selected_status[idx] = True
                N_t[idx] += 1
        elif t == np.floor(num_clients / num_selected):
            for idx in range(t * num_selected, num_clients):
                selected_status[idx] = True
                N_t[idx] += 1
            remaining_selections = num_selected * (t + 1) - num_clients
            if remaining_selections > 0:
                unselected_indices = list(range(0, t * num_selected))
                selected_indices_subset = np.random.choice(
                    unselected_indices, size=remaining_selections, replace=False
                )
                for idx in selected_indices_subset:
                    selected_status[idx] = True
                    N_t[idx] += 1
        else:
            # do UCB selection
            selected_indices = topk(UCB, num_selected)
            for idx in selected_indices:
                selected_status[idx] = True
                N_t[idx] += 1
        # uniform random
        client_states = []
        weights = []

        for idx, client in enumerate(clients):
            if selected_status[idx]:
                # perform descent at client
                client_state = client.train(
                    server.model,
                    criterion=fed_avg_criterion(),
                    E=E,
                    B=B,
                    learning_rate=learning_rate,
                    momentum=momentum,
                )
                weight = client.length  # number of data points at client
                client_states.append(client_state)
                weights.append(weight)

        # compute shapley values for each client BEFORE updating server model

        # print('starting MC')
        # server.model_evaluations = 0
        # shapley_values = server.shapley_values_mc(
        #     fed_avg_criterion(), client_states, weights
        # )
        # print(f'SV = {shapley_values}')
        # print(f'server evaluations = {server.model_evaluations}')

        # print("starting TMC")
        # server.model_evaluations = 0
        # shapley_values_tmc = server.shapley_values_tmc(
        #     fed_avg_criterion(), deepcopy(client_states), deepcopy(weights)
        # )
        # print(f"server evaluations = {server.model_evaluations}")
        # # print(f"SV = {shapley_values_tmc}")
        # num_model_evaluations["tmc"].append(server.model_evaluations)

        print("starting GTG")
        server.model_evaluations = 0
        shapley_values_gtg = server.shapley_values_gtg(
            fed_avg_criterion(), client_states, weights
        )
        print(f"server evaluations = {server.model_evaluations}")
        # print(f"SV = {shapley_values_gtg}")
        num_model_evaluations["gtg"].append(server.model_evaluations)

        # print("starting True")
        # server.model_evaluations = 0
        # shapley_values_true = server.shapley_values_true(
        #     fed_avg_criterion(), client_states, weights
        # )
        # print(f"server evaluations = {server.model_evaluations}")
        # # print(f"SV = {shapley_values_true}")
        # num_model_evaluations["true"].append(server.model_evaluations)

        sv_rounds["gtg"].append(shapley_values_gtg)
        # sv_rounds["tmc"].append(shapley_values_tmc)
        # sv_rounds["true"].append(shapley_values_true)

        shapley_values = shapley_values_gtg
        # shapley_values = shapley_values / np.sum(shapley_values)

        # update server model
        server.aggregate(client_states, weights)
        test_acc_now = server.accuracy()
        train_acc_now = np.sum(
            [
                client_weights[i] * clients[i].accuracy_(server.model)
                for i in range(num_clients)
            ]
        )
        train_loss_now = np.sum(
            [
                client_weights[i] * clients[i].loss(server.model, fed_avg_criterion())
                for i in range(num_clients)
            ]
        )
        val_loss_now = server.val_loss(server.model, fed_avg_criterion())
        test_loss_now = server.test_loss(fed_avg_criterion())

        train_acc.append(train_acc_now)
        test_acc.append(test_acc_now)
        train_loss.append(train_loss_now)
        val_loss.append(val_loss_now)
        test_loss.append(test_loss_now)

        # compute UCB for next round of selections
        selections = [0 for i in range(num_clients)]
        counter = 0
        for i in range(num_clients):
            if selected_status[i]:
                SV[i] = ((N_t[i] - 1) * SV[i] + shapley_values[counter]) / N_t[i]
                counter += 1
                selections[i] = 1
            UCB[i] = SV[i] + beta * np.sqrt(np.log(t + 1) / N_t[i])
        shapley_values_T.append(deepcopy(SV))
        ucb_values_T.append(deepcopy(UCB))
        selections_T.append(deepcopy(selections))
        draws_T.append(deepcopy(N_t))
        log_dict = {
            "train_accuracy": train_acc_now,
            "test_accuracy": test_acc_now,
            "train_loss": train_loss_now,
            "val_loss": val_loss_now,
            "test_loss": test_loss_now,
        }

        for i in range(num_clients):
            log_dict[f"shapley_value_{i}"] = SV[i]
            log_dict[f"selection_{i}"] = selections[i]

        if logging == True:
            wandb.log(log_dict)

        # if t % 10 == 0:
        #     sns.heatmap(selections_T).set(title="selections")
        #     plt.show()
        #     sns.heatmap(shapley_values_T).set(title="SV")
        #     plt.show()

    if logging == True:
        wandb.finish()

    return (
        test_acc,
        train_acc,
        train_loss,
        val_loss,
        test_loss,
        selections_T,
        shapley_values_T,
        sv_rounds,
        num_model_evaluations,
        ucb_values_T,
    )


def sfedavg_run(
    clients,
    server,
    select_fraction,
    T,
    alpha,
    beta,
    random_seed=0,
    E=5,
    B=10,
    learning_rate=0.01,
    momentum=0.5,
    logging=False,
):
    clients = deepcopy(clients)
    client_weights = np.array([client.length for client in clients])
    client_weights = client_weights / np.sum(client_weights)
    server = deepcopy(server)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_clients = len(clients)
    num_selected = int(np.ceil(select_fraction * num_clients))

    test_acc = []
    train_acc = []
    train_loss = []
    val_loss = []
    test_loss = []

    shapley_values_T = []
    selections_T = []
    Phi_T = []
    draws_T = []

    N_t = [0 for i in range(num_clients)]
    Phi = [1 / (num_clients) for i in range(num_clients)]
    SV = [0 for i in range(num_clients)]
    for t in tqdm(range(T)):
        # select clients to transmit weights to
        # initially sample every client atleast once
        selected_status = [False for i in range(num_clients)]
        # do Game of Gradients Selection
        all_indices = list(range(num_clients))
        probs = np.exp(np.array(Phi))
        probs = probs / np.sum(probs)
        selected_indices = np.random.choice(
            all_indices, size=num_selected, replace=False, p=probs
        )
        for idx in selected_indices:
            selected_status[idx] = True
            N_t[idx] += 1
        # uniform random
        client_states = []
        weights = []

        for idx, client in enumerate(clients):
            if selected_status[idx]:
                # perform descent at client
                client_state = client.train(
                    server.model,
                    criterion=fed_avg_criterion(),
                    E=E,
                    B=B,
                    learning_rate=learning_rate,
                    momentum=momentum,
                )
                weight = client.length  # number of data points at client
                weight /= probs[idx]  # for unbiased averaging
                client_states.append(client_state)
                weights.append(weight)

        # compute shapley values for each client BEFORE updating server model
        # shapley_values = server.shapley_values_mc(
        #     fed_avg_criterion(), client_states, weights
        # )
        # shapley_values = server.shapley_values_tmc(
        #     fed_avg_criterion(), client_states, weights
        # )
        shapley_values = server.shapley_values_gtg(
            fed_avg_criterion(), client_states, weights
        )
        # update server model
        server.aggregate(client_states, weights)
        test_acc_now = server.accuracy()
        train_acc_now = np.sum(
            [
                client_weights[i] * clients[i].accuracy_(server.model)
                for i in range(num_clients)
            ]
        )
        train_loss_now = np.sum(
            [
                client_weights[i] * clients[i].loss(server.model, fed_avg_criterion())
                for i in range(num_clients)
            ]
        )
        val_loss_now = server.val_loss(server.model, fed_avg_criterion())
        test_loss_now = server.test_loss(fed_avg_criterion())

        train_acc.append(train_acc_now)
        test_acc.append(test_acc_now)
        train_loss.append(train_loss_now)
        val_loss.append(val_loss_now)
        test_loss.append(test_loss_now)

        # compute Phi for next round of selections
        selections = [0 for i in range(num_clients)]
        counter = 0
        # defined as function parameters now
        # alpha = 0.75
        # beta = 0.25
        for i in range(num_clients):
            if selected_status[i]:
                SV[i] = ((N_t[i] - 1) * SV[i] + shapley_values[counter]) / N_t[i]
                counter += 1
                selections[i] = 1
                Phi[i] = alpha * Phi[i] + beta * SV[i]
        shapley_values_T.append(deepcopy(SV))
        Phi_T.append(deepcopy(Phi))
        selections_T.append(deepcopy(selections))
        draws_T.append(deepcopy(N_t))

        log_dict = {
            "train_accuracy": train_acc_now,
            "test_accuracy": test_acc_now,
            "train_loss": train_loss_now,
            "val_loss": val_loss_now,
            "test_loss": test_loss_now,
        }

        for i in range(num_clients):
            log_dict[f"shapley_value_{i}"] = SV[i]
            log_dict[f"selection_{i}"] = selections[i]
        if logging == True:
            wandb.log(log_dict)

    if logging == True:
        wandb.finish()

    return (
        test_acc,
        train_acc,
        train_loss,
        val_loss,
        test_loss,
        selections_T,
        shapley_values_T,
    )

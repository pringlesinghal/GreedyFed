import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from utilities import *
from dshap import convergenceTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aggregator_update(client_states, sent_status, model):
    # find updated model
    active_clients = np.sum(sent_status)

    model_state = model.state_dict()
    for key in model_state.keys():
        model_state[key] -= model_state[key]  # set to zero
    # find updated model by averaging weights from selected clients
    for idx, status in enumerate(sent_status):
        if status:
            for key in model_state.keys():
                model_state[key] += (1 / active_clients) * client_states[idx][key]

    model.load_state_dict(model_state)


def aggregator_update_shapley(
    client_values,
    alpha,
    beta,
    client_states,
    sent_status,
    model,
    val_data,
    criterion,
    device,
):
    # find updated model
    active_clients = np.sum(sent_status)

    model_state = model.state_dict()
    init_state = deepcopy(model_state)
    for key in model_state.keys():
        model_state[key] -= model_state[key]  # set to zero
    # find updated model by averaging weights from selected clients
    for idx, status in enumerate(sent_status):
        if status:
            for key in model_state.keys():
                model_state[key] += (1 / active_clients) * client_states[idx][key]

    model.load_state_dict(model_state)

    init_model = deepcopy(model)
    init_model.load_state_dict(init_state)

    # calculate shapley values of chosen clients
    active_client_indices = np.where(sent_status)[0]
    final_shapley_values = []
    for active_client_idx in active_client_indices:
        remaining_client_indices = list(
            set(active_client_indices) - set([active_client_idx])
        )
        T = 50
        t = 0
        shapley_values = []
        avg_shapley_value = 0
        while t < T:
            if convergenceTest(shapley_values):
                break
            t += 1
            loss_i = 0
            loss_f = 0
            # pick a size for the random subset
            subset_sizes = [i for i in range(len(remaining_client_indices))]
            subset_size = np.random.choice(subset_sizes, size=1)[0]
            # select a random subset of gradients that were transmitted
            chosen_client_indices = np.random.choice(
                remaining_client_indices, size=subset_size, replace=False
            )
            # compute updated model without active_client_idx
            if subset_size > 0:
                model_S = deepcopy(init_model)
                client_states_S = []
                sent_status_S = []
                for idx, state in enumerate(client_states):
                    if idx in chosen_client_indices:
                        client_states_S.append(state)
                        sent_status_S.append(True)
                aggregator_update(
                    client_states=client_states_S,
                    sent_status=sent_status_S,
                    model=model_S,
                )

                with torch.no_grad():
                    scores = model_S(val_data.data)
                    loss = criterion(scores, val_data.targets)
            else:
                with torch.no_grad():
                    scores = init_model(val_data.data)
                    loss = criterion(scores, val_data.targets)
            loss_i = loss

            # compute updated model with active_client_idx
            chosen_client_indices = list(chosen_client_indices)
            chosen_client_indices.append(active_client_idx)

            model_Sux = deepcopy(init_model)
            client_states_Sux = []
            sent_status_Sux = []
            for idx, state in enumerate(client_states):
                if idx in chosen_client_indices:
                    client_states_Sux.append(state)
                    sent_status_Sux.append(True)
            aggregator_update(
                client_states=client_states_Sux,
                sent_status=sent_status_Sux,
                model=model_Sux,
            )
            with torch.no_grad():
                scores = model_Sux(val_data.data)
                loss = criterion(scores, val_data.targets)
            loss_f = loss
            # calculate shapley value for client_idx as difference in val loss
            shapley_value_t = loss_i - loss_f
            # calculate average shapley value and append to list
            avg_shapley_value = (
                avg_shapley_value * len(shapley_values) + shapley_value_t
            ) / (len(shapley_values) + 1)
            shapley_values.append(avg_shapley_value)

        final_shapley_values.append(avg_shapley_value)

    sv_updates = []
    counter = 0

    for idx, client_value in enumerate(client_values):
        if idx in active_client_indices:
            # to match order of magnitude
            client_value = (
                alpha * client_value + beta * final_shapley_values[counter] * 10
            )
            counter += 1
        sv_updates.append(client_value)
    return torch.Tensor(sv_updates)


def aggregator_update_ucb(
    communication_round,
    beta,
    sv,
    nk,
    ucb,
    client_states,
    sent_status,
    model,
    val_data,
    criterion,
    device,
):
    # find updated model
    active_clients = np.sum(sent_status)

    model_state = model.state_dict()
    init_state = deepcopy(model_state)
    for key in model_state.keys():
        model_state[key] -= model_state[key]  # set to zero
    # find updated model by averaging weights from selected clients
    for idx, status in enumerate(sent_status):
        if status:
            for key in model_state.keys():
                model_state[key] += (1 / active_clients) * client_states[idx][key]

    model.load_state_dict(model_state)

    init_model = deepcopy(model)
    init_model.load_state_dict(init_state)

    # calculate shapley values of chosen clients
    active_client_indices = np.where(sent_status)[0]
    final_shapley_values = []
    for active_client_idx in active_client_indices:
        remaining_client_indices = list(
            set(active_client_indices) - set([active_client_idx])
        )
        T = 50
        t = 0
        shapley_values = []
        avg_shapley_value = 0
        while t < T:
            if convergenceTest(shapley_values):
                break
            t += 1
            loss_i = 0
            loss_f = 0
            # pick a size for the random subset
            subset_sizes = [i for i in range(len(remaining_client_indices))]
            subset_size = np.random.choice(subset_sizes, size=1)[0]
            # select a random subset of gradients that were transmitted
            chosen_client_indices = np.random.choice(
                remaining_client_indices, size=subset_size, replace=False
            )
            # compute updated model without active_client_idx
            if subset_size > 0:
                model_S = deepcopy(init_model)
                client_states_S = []
                sent_status_S = []
                for idx, state in enumerate(client_states):
                    if idx in chosen_client_indices:
                        client_states_S.append(state)
                        sent_status_S.append(True)
                aggregator_update(
                    client_states=client_states_S,
                    sent_status=sent_status_S,
                    model=model_S,
                )

                with torch.no_grad():
                    scores = model_S(val_data.data)
                    loss = criterion(scores, val_data.targets)
            else:
                with torch.no_grad():
                    scores = init_model(val_data.data)
                    loss = criterion(scores, val_data.targets)
            loss_i = loss

            # compute updated model with active_client_idx
            chosen_client_indices = list(chosen_client_indices)
            chosen_client_indices.append(active_client_idx)

            model_Sux = deepcopy(init_model)
            client_states_Sux = []
            sent_status_Sux = []
            for idx, state in enumerate(client_states):
                if idx in chosen_client_indices:
                    client_states_Sux.append(state)
                    sent_status_Sux.append(True)
            aggregator_update(
                client_states=client_states_Sux,
                sent_status=sent_status_Sux,
                model=model_Sux,
            )
            with torch.no_grad():
                scores = model_Sux(val_data.data)
                loss = criterion(scores, val_data.targets)
            loss_f = loss
            # calculate shapley value for client_idx as difference in val loss
            shapley_value_t = loss_i - loss_f
            # calculate average shapley value and append to list
            avg_shapley_value = (
                avg_shapley_value * len(shapley_values) + shapley_value_t
            ) / (len(shapley_values) + 1)
            shapley_values.append(avg_shapley_value)

        final_shapley_values.append(avg_shapley_value)

    counter = 0
    for idx, shapley_value in enumerate(sv):
        if idx in active_client_indices:
            # to match order of magnitude
            shapley_value = (
                (communication_round * shapley_value) + final_shapley_values[counter]
            ) / (communication_round + 1)
            # moving average with respect to number of communication rounds
            counter += 1
            nk[idx] += 1
        sv[idx] = shapley_value
        if nk[idx] > 0:
            ucb[idx] = shapley_value + beta * np.sqrt(
                np.log(communication_round + 1) / nk[idx]
            )

    return sv, nk, ucb

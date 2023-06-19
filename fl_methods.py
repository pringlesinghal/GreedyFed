import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from utilities import *
from dshap import convergenceTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aggregator_update(client_gradients, sent_status, model, optimiser):
    # find updated model
    num_clients = len(sent_status)
    active_clients = np.sum(sent_status)
    weights = [0 for _ in range(num_clients)]
    for idx, status in enumerate(sent_status):
        # uniform weights to active clients
        weights[idx] = (
            active_clients and status and 1 / (active_clients)
        )  # to handle division by zero

    weights = torch.Tensor(weights).to(device)
    for i in client_gradients:
        if i is not None:
            # make gradients variable different from None
            model.set_gradients(i)
            break

    optimiser.zero_grad(set_to_none=False)  # clear aggregator gradients to zero
    agg_gradient = model.gradients()
    # calculate gradients for all parameters using all clients
    for i in range(num_clients):
        if sent_status[i]:  # otherwise we skip
            # client_gradients[i], tuple with gradients for all weights and biases
            for param_idx in range(len(agg_gradient)):
                agg_gradient[param_idx] += weights[i] * client_gradients[i][param_idx]

    # update aggregator gradient and then update weights
    model.set_gradients(agg_gradient)
    optimiser.step()


def aggregator_update_shapley(
    client_values,
    client_gradients,
    sent_status,
    model,
    optimiser,
    val_data,
    criterion,
    device,
):
    # update shapley values for clients with sent status True
    # use val data for calculation

    # find updated model
    num_clients = len(sent_status)
    active_clients = np.sum(sent_status)
    weights = [0 for _ in range(num_clients)]
    for idx, status in enumerate(sent_status):
        # uniform weights to active clients
        weights[idx] = (
            active_clients and status and 1 / (active_clients)
        )  # to handle division by zero

    weights = torch.Tensor(weights).to(device)
    for i in client_gradients:
        if i is not None:
            # make gradients variable different from None
            model.set_gradients(i)
            break

    optimiser.zero_grad(set_to_none=False)  # clear aggregator gradients to zero
    agg_gradient = model.gradients()

    init_model = deepcopy(model)
    init_model.load_state_dict(model.state_dict())

    # calculate gradients for all parameters using all clients
    for i in range(num_clients):
        if sent_status[i]:  # otherwise we skip
            # client_gradients[i], tuple with gradients for all weights and biases
            for param_idx in range(len(agg_gradient)):
                agg_gradient[param_idx] += weights[i] * client_gradients[i][param_idx]

    # update aggregator gradient and then update weights
    model.set_gradients(agg_gradient)
    optimiser.step()

    # calculate shapley values using init model
    active_client_indices = np.where(sent_status)[0]
    for active_client_idx in active_client_indices:
        init_client_value = client_values[active_client_idx]
        remaining_client_indices = list(
            set(active_client_indices) - set([active_client_idx])
        )
        T = 50
        t = 0
        shapley_values = []
        avg_shapley_value = 0
        while t < T:
            if convergenceTest(shapley_values):
                print("Converged")
                break
            t += 1
            loss_i = 0
            loss_f = 0
            # pick a size for the random subset
            subset_sizes = [i for i in range(len(remaining_client_indices))]
            subset_size = np.random.choice(subset_sizes, size=1)[0]
            # select a random subset of gradients that were transmitted
            chosen_gradient_indices = np.random.choice(
                remaining_client_indices, size=subset_size, replace=False
            )
            # compute updated model without active_client_idx
            if subset_size > 0:
                model_S = deepcopy(init_model).to(device)
                model_S.load_state_dict(model.state_dict())
                optimiser_S = optim.Adam(model_S.parameters())
                optimiser_S.load_state_dict(optimiser.state_dict())

                optimiser_S.zero_grad(set_to_none=False)
                agg_gradient = model_S.gradients()

                for idx in chosen_gradient_indices:
                    for param_idx in range(len(model_S.parameters())):
                        agg_gradient[param_idx] += (1 / subset_size) * client_gradients[
                            idx
                        ][param_idx]
                model_S.set_gradients(agg_gradient)
                optimiser_S.step()
                with torch.no_grad():
                    scores = model_S(val_data.data)
                    loss = criterion(scores, val_data.targets)
            else:
                with torch.no_grad():
                    init_model = init_model.cpu()
                    scores = init_model(val_data.data.cpu())
                    loss = criterion(scores, val_data.targets.cpu())
            loss_i = loss

            # compute updated model with active_client_idx
            chosen_gradient_indices.append(active_client_idx)

            model_Sux = deepcopy(init_model).to(device)
            model_Sux.load_state_dict(model.state_dict())
            optimiser_Sux = optim.Adam(model_Sux.parameters())
            optimiser_Sux.load_state_dict(optimiser.state_dict())
            optimiser_Sux.zero_grad(set_to_none=False)
            agg_gradient = model_Sux.gradients()

            for idx in chosen_gradient_indices:
                for param_idx in range(len(model_Sux.parameters())):
                    agg_gradient[param_idx] += (1 / subset_size) * client_gradients[
                        idx
                    ][param_idx]
            model_Sux.set_gradients(agg_gradient)
            optimiser_Sux.step()
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

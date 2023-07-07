import torch
import torch.optim as optim
import numpy as np

from copy import deepcopy
from itertools import chain, combinations
from math import comb

from utils import convergenceTest


class Server:
    def __init__(self, model, val_data, val_targets, test_data, test_targets, device):
        self.model = deepcopy(model).to(device)
        self.val_data = val_data.to(device=device)
        self.val_targets = val_targets.to(device=device)
        self.test_data = test_data.to(device=device)
        self.test_targets = test_targets.to(device=device)
        self.length = len(test_data)
        self.device = device

        # to keep track of the number of model validation loss evaluations for shapley algorithms
        self.model_evaluations = 0

    def aggregate(self, client_states, weights=None):
        """
        client_states - list of client states
        weights - weights for averaging (uniform by default)

        updates server model by performing weighted averaging
        """
        model = self.aggregate_(client_states, weights)
        self.model.load_state_dict(model.state_dict())

    def aggregate_(self, client_states, weights=None):
        """
        does not modify the server model
        only returns the updated model
        """
        if len(client_states) == 0:
            return deepcopy(self.model).to(device=self.device)
        if weights is None:
            # uniform weights by default
            weights = [1 / len(client_states)] * len(client_states)
        weights = np.array(weights)
        wtsum = np.sum(weights)
        weights = weights / wtsum  # normalize weights
        # initialise model parameters to zero
        model_state = deepcopy(self.model).state_dict()
        for key in model_state.keys():
            model_state[key] -= model_state[key]
        # find updated model - weighted averaging
        for idx, client_state in enumerate(client_states):
            for key in model_state.keys():
                model_state[key] += weights[idx] * client_state[key]
        model = deepcopy(self.model).to(device=self.device)
        model.load_state_dict(model_state)
        return model

    def shapley_values_mc(self, criterion, client_states, weights=None):
        """
        client_states - list of client states
        weights - weights for averaging (uniform by default)

        computes shapley values for the client updates on validation dataset
        """
        self.model_evaluations = 0
        if weights is None:
            # uniform weights by default
            weights = [1 / len(client_states)] * len(client_states)
        weights = np.array(weights)
        wtsum = np.sum(weights)
        weights = weights / wtsum  # normalize weights

        num_clients = len(client_states)
        T = 50

        shapley_values = [[0] for i in range(num_clients)]

        for idx in range(num_clients):
            # compute shapley value of idx client
            """
            until convergence:
                sample a subset size k
                sample subset of size k of clients (except idx)
                compute updated model with this subset of clients
                compute loss of updated model on validation set
                compute another updated model with the idx client included
                compute loss of updated model on validation set
                compute difference between losses of the two models
                average losses over subsets to compute the shapley value of idx client
            """
            t = 0
            converged = False
            remaining_clients = [i for i in range(num_clients) if i != idx]
            while t < T:
                subset_size = np.random.choice(list(range(num_clients - 1)), size=1)[0]
                subset = np.random.choice(
                    remaining_clients, size=subset_size, replace=False
                )
                client_states_subset = [client_states[i] for i in subset]
                weights_subset = [weights[i] for i in subset]
                model_subset = self.aggregate_(client_states_subset, weights_subset)
                loss_subset = self.val_loss(model_subset, criterion)
                value_subset = 1 - loss_subset

                client_states_subset.append(client_states[idx])
                weights_subset.append(weights[idx])
                model_subset_with_idx = self.aggregate_(
                    client_states_subset, weights_subset
                )
                loss_subset_with_idx = self.val_loss(model_subset_with_idx, criterion)
                value_subset_with_idx = 1 - loss_subset_with_idx

                utility_gain = value_subset_with_idx - value_subset
                prev_avg = shapley_values[idx][-1]
                new_avg = (t * prev_avg + utility_gain) / (t + 1)
                shapley_values[idx].append(new_avg)
                if convergenceTest(shapley_values[idx]):
                    converged = True
                t += 1
        if converged == False:
            print("SV not converged in MC")
        final_shapley_values = [shapley_values[i][-1] for i in range(num_clients)]
        return final_shapley_values

    def shapley_values_tmc(self, criterion, client_states, weights=None):
        """
        client_states - list of client states
        weights - weights for averaging (uniform by default)

        computes shapley values for the client updates on validation dataset
        """
        self.model_evaluations = 0
        if weights is None:
            # uniform weights by default
            weights = [1 / len(client_states)] * len(client_states)
        weights = np.array(weights)
        wtsum = np.sum(weights)
        weights = weights / wtsum  # normalize weights

        num_clients = len(client_states)

        shapley_values = [[0] for i in range(num_clients)]
        converged = False

        T = 50 * num_clients
        t = 0
        threshold = 1e-4
        v_init = 1 - self.val_loss(self.model, criterion)  # initial server model loss
        model_final = self.aggregate_(client_states, weights)
        v_final = 1 - self.val_loss(model_final, criterion)  # final server model loss
        while not converged and (t < T):
            t += 1
            client_permutation = np.random.permutation(num_clients)
            v_j = v_init
            for j in range(num_clients):
                if np.abs(v_final - v_j) < threshold:
                    v_jplus1 = v_j
                else:
                    subset = client_permutation[: (j + 1)]
                    client_states_subset = [client_states[i] for i in subset]
                    weights_subset = [weights[i] for i in subset]
                    model_subset = self.aggregate_(client_states_subset, weights_subset)
                    v_jplus1 = 1 - self.val_loss(model_subset, criterion)

                phi_old = shapley_values[client_permutation[j]][-1]
                phi_new = ((t - 1) * phi_old + (v_jplus1 - v_j)) / t
                shapley_values[client_permutation[j]].append(phi_new)
                v_j = v_jplus1

            flag = True
            shapley_avg = np.mean(shapley_values, axis=0)
            if not convergenceTest(shapley_avg):
                flag = False
            if flag:
                converged = True
        if converged == False:
            print("not converged in SV TMC")
        final_shapley_values = [shapley_values[i][-1] for i in range(num_clients)]
        return final_shapley_values

    def shapley_values_gtg(self, criterion, client_states, weights=None):
        """
        client_states - list of client states
        weights - weights for averaging (uniform by default)

        computes shapley values for the client updates on validation dataset
        """
        self.model_evaluations = 0
        if weights is None:
            # uniform weights by default
            weights = [1 / len(client_states)] * len(client_states)
        weights = np.array(weights)
        wtsum = np.sum(weights)
        weights = weights / wtsum  # normalize weights

        num_clients = len(client_states)

        shapley_values = [[0] for i in range(num_clients)]
        converged = False

        T = 50 * num_clients
        t = 0
        threshold = 1e-4
        v_init = 1 - self.val_loss(self.model, criterion)  # initial server model loss
        model_final = self.aggregate_(client_states, weights)
        v_final = 1 - self.val_loss(model_final, criterion)  # final server model loss
        if np.abs(v_final - v_init) < threshold:
            # between round truncation
            print(
                f"between round truncation: {v_final} - {v_init} = {np.abs(v_final - v_init)}"
            )

            epsilon = 1e-9
            return [epsilon for i in range(num_clients)]

        while not converged and (t < T):
            for client_idx in range(num_clients):
                t += 1
                client_permutation = np.concatenate(
                    (
                        np.array([client_idx]),
                        np.random.permutation(
                            [i for i in range(num_clients) if i != client_idx]
                        ),
                    )
                ).astype(int)
                v_j = v_init
                for j in range(num_clients):
                    if np.abs(v_final - v_j) < threshold:
                        v_jplus1 = v_j
                    else:
                        subset = client_permutation[: (j + 1)]
                        client_states_subset = [client_states[i] for i in subset]
                        weights_subset = [weights[i] for i in subset]
                        model_subset = self.aggregate_(
                            client_states_subset, weights_subset
                        )
                        v_jplus1 = 1 - self.val_loss(model_subset, criterion)

                    phi_old = shapley_values[client_permutation[j]][-1]
                    phi_new = ((t - 1) * phi_old + (v_jplus1 - v_j)) / t
                    shapley_values[client_permutation[j]].append(phi_new)
                    v_j = v_jplus1

            flag = True
            shapley_avg = np.mean(shapley_values, axis=0)
            if not convergenceTest(shapley_avg):
                flag = False
            if flag:
                converged = True
        if converged == False:
            print("not converged in SV GTG")
        final_shapley_values = [shapley_values[i][-1] for i in range(num_clients)]
        return final_shapley_values

    def shapley_values_true(self, criterion, client_states, weights=None):
        """
        client_states - list of client states
        weights - weights for averaging (uniform by default)

        computes shapley values for the client updates on validation dataset
        """
        self.model_evaluations = 0

        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return list(
                chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
            )

        if weights is None:
            # uniform weights by default
            weights = [1 / len(client_states)] * len(client_states)
        weights = np.array(weights)
        wtsum = np.sum(weights)
        weights = weights / wtsum  # normalize weights

        num_clients = len(client_states)
        client_subsets = powerset(range(num_clients))
        subset_utilities = {i: 0 for i in client_subsets}
        shapley_values = [[0] for i in range(num_clients)]

        for subset in client_subsets:
            client_states_subset = [client_states[i] for i in subset]
            weights_subset = [weights[i] for i in subset]
            model_subset = self.aggregate_(client_states_subset, weights_subset)
            loss_subset = self.val_loss(model_subset, criterion)
            subset_utilities[subset] = 1 - loss_subset

        for subset in client_subsets:
            for idx in range(num_clients):
                L = len(subset)  # subset size
                if idx in subset:
                    nck = comb(num_clients - 1, L - 1)
                    prev_val = shapley_values[idx][-1]
                    new_val = prev_val + subset_utilities[subset] / (num_clients * nck)
                    shapley_values[idx].append(new_val)
                else:
                    nck = comb(num_clients - 1, L)
                    prev_val = shapley_values[idx][-1]
                    new_val = prev_val - subset_utilities[subset] / (num_clients * nck)
                    shapley_values[idx].append(new_val)

        final_shapley_values = [shapley_values[i][-1] for i in range(num_clients)]
        return final_shapley_values

    def test_loss(self, criterion):
        """
        criterion - loss function (model, data, targets)

        computes loss on test set with the server model
        """
        self.model.eval()
        with torch.no_grad():
            loss = criterion(self.model, self.test_data, self.test_targets)
        self.model.train()
        return float(loss.cpu())

    def accuracy(self):
        """
        test accuracy
        """
        self.model.eval()
        with torch.no_grad():
            scores = self.model(self.test_data)
            _, predictions = scores.max(1)
            num_correct = torch.sum(predictions == self.test_targets)
            total = self.length
            accuracy = num_correct / total
        self.model.train()
        return float(accuracy.cpu())

    def val_loss(self, model, criterion):
        """
        model
        criterion - loss function (model, data, targets)

        computes loss on validation set with the given model
        """
        model.eval()
        with torch.no_grad():
            loss = criterion(model, self.val_data, self.val_targets)
        model.train()
        self.model_evaluations += 1
        return float(loss.cpu())

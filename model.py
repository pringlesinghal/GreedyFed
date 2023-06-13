import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.hidden_dims = [50, 25]
        self.fc1 = nn.Linear(input_dim, self.hidden_dims[0])
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.fc3 = nn.Linear(self.hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def gradients(self):
        return [
            self.fc1.weight.grad,
            self.fc1.bias.grad,
            self.fc2.weight.grad,
            self.fc2.bias.grad,
            self.fc3.weight.grad,
            self.fc3.bias.grad,
        ]

    def set_gradients(self, gradients):
        self.fc1.weight.grad = gradients[0]
        self.fc1.bias.grad = gradients[1]
        self.fc2.weight.grad = gradients[2]
        self.fc2.bias.grad = gradients[3]
        self.fc3.weight.grad = gradients[4]
        self.fc3.bias.grad = gradients[5]

    def set_weights(self, state):
        net_state = self.state_dict()
        for param in net_state.keys():
            net_state[param] = state[param]
        self.load_state_dict(net_state)

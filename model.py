import torch
import torch.nn as nn
import numpy as np
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


class CNN(nn.Module):
    def __init__(self, in_channels, input_w, input_h, output_dim):
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.input_w = input_w
        self.input_h = input_h
        self.output_dim = output_dim

        out_channels_1 = 8
        kernel_size_1 = 4
        padding_1 = 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels=out_channels_1,
            kernel_size=kernel_size_1,
            padding=padding_1,
        )
        output_1_w = input_w - kernel_size_1 + 2 * padding_1 + 1
        output_1_h = input_h - kernel_size_1 + 2 * padding_1 + 1

        kernel_size_2 = 2
        stride_2 = 2
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_2, stride=stride_2)
        output_2_w = np.floor((output_1_w - kernel_size_2) / stride_2) + 1
        output_2_h = np.floor((output_1_h - kernel_size_2) / stride_2) + 1

        in_channels_3 = out_channels_1
        out_channels_3 = 8
        kernel_size_3 = 4
        padding_3 = 1
        self.conv2 = nn.Conv2d(
            in_channels_3, out_channels_3, kernel_size=kernel_size_3, padding=padding_1
        )
        output_3_w = output_2_w - kernel_size_3 + 2 * padding_3 + 1
        output_3_h = output_2_h - kernel_size_3 + 2 * padding_3 + 1

        # after pooling again
        output_4_w = np.floor((output_3_w - kernel_size_2) / stride_2) + 1
        output_4_h = np.floor((output_3_h - kernel_size_2) / stride_2) + 1
        input_size_4 = int(output_4_h * output_4_w * out_channels_3)
        self.fc = nn.Linear(input_size_4, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

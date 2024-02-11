"""
Jet Substructure Models used in the LogicNets paper
"""

import torch.nn as nn
import torch

class JSC_Toy(nn.Module):
    def __init__(self, info):
        super(JSC_Toy, self).__init__()
        self.param = nn.Parameter(torch.rand(3, 4))
        self.linear = nn.Linear(16, 5)

    def forward(self, x):
        return torch.topk(torch.sum(
            self.linear(x + self.linear.weight).relu(), dim=-1), 3)



class JSC_Tiny(nn.Module):
    def __init__(self, info):
        super(JSC_Tiny, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 5),  # linear              # 2
            # nn.BatchNorm1d(5),  # output_quant       # 3
            nn.ReLU(5),  # 4
        )

    def forward(self, x):
        return self.seq_blocks(x)


class JSC_S(nn.Module):
    def __init__(self, info):
        super(JSC_S, self).__init__()
        self.config = info
        self.num_features = self.config.num_features
        self.num_classes = self.config.num_classes
        hidden_layers = [64, 32, 32, 32]
        self.num_neurons = [self.num_features] + hidden_layers + [self.num_classes]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            layer = []
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                in_act = nn.ReLU()
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [bn_in, in_act, fc, bn, out_act]
            elif i == len(self.num_neurons) - 1:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, bn, out_act]
            else:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, out_act]
            layer_list = layer_list + layer
        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x

class JSC_Lab1(nn.Module):
    def __init__(self, info):
        super(JSC_Lab1, self).__init__()
        self.config = info
        self.num_features = self.config.num_features
        self.num_classes = self.config.num_classes
        hidden_layers = [32]
        self.num_neurons = [self.num_features] + hidden_layers + [self.num_classes]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            layer = []
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                in_act = nn.ReLU()
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [bn_in, in_act, fc, bn, out_act]
            elif i == len(self.num_neurons) - 1:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, bn, out_act]
            else:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, out_act]
            layer_list = layer_list + layer
        self.module_list = nn.ModuleList(layer_list)
        
        self.bn1 = nn.BatchNorm1d(self.num_features)
        self.bn2 = nn.BatchNorm1d(16)
        # self.bn3 = nn.BatchNorm1d(32)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.num_features * 2, 16)
        # self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(16, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(x)
        x = x.view(x.size(0), 1, -1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(self.relu(self.bn2(x)))
        x = self.fc3(self.relu(self.bn2(x)))

        return x
    
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self, info):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)


# Getters ------------------------------------------------------------------------------
def get_jsc_toy(info):
    # TODO: Tanh is not supported by mase yet
    return JSC_Toy(info)


def get_jsc_tiny(info):
    return JSC_Tiny(info)


def get_jsc_s(info):
    return JSC_S(info)

def get_jsc_lab1(info):
    return JSC_Lab1(info)

def get_jsc_three_linear(info):
    return JSC_Three_Linear_Layers(info)
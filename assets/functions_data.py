from torch_geometric.transforms import RandomNodeSplit as masking
import utils
from model.GAT import *
from model.GraphSAGE import *
from model.GCN import *
import torch
import torch.nn as nn
import utils


def create_graph(data_raw, edge_data, target_data, train_split=0.2, val_split=0.1, test_split=0.6):
    data_encoded, _ = utils.encode_data(data_raw, light=False)
    g = utils.construct_graph(
        target_data, edge_data, data_encoded=data_encoded, light=False)

    msk = masking(split="train_rest", num_splits=1,
                  num_train_per_class=train_split, num_val=val_split, num_test=test_split)
    g = msk(g)
    return g


def submit(model, data, epochs, learning_rate, num_features, hidden_channels, num_classes):
    if model == 'GCN':
        gcn = GCN(num_features, hidden_channels,  num_classes)
        criterion = nn.CrossEntropyLoss()
        train_accuracies, val_accuracies, test_accuracy = utils.train(
            gcn, data, epochs, learning_rate)
    elif model == 'GAT':
        gAT = GAT(num_features, hidden_channels,  num_classes)
        criterion = nn.CrossEntropyLoss()
        train_accuracies, val_accuracies, test_accuracy = utils.train(
            gAT, data, epochs, learning_rate)
    else:
        gSEA = GraphSAGE(num_features, hidden_channels,  num_classes)
        criterion = nn.CrossEntropyLoss()
        train_accuracies, val_accuracies, test_accuracy = utils.train(
            gSEA, data, epochs, learning_rate)

    return train_accuracies, val_accuracies, test_accuracy

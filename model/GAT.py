import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
import utils
import numpy as np


class GAT(torch.nn.Module):
    def __init__(self, num_of_feat, f, num_of_label):
        super(GAT, self).__init__()

        self.conv1 = GATv2Conv(num_of_feat, f)

        self.conv2 = GATv2Conv(f, num_of_label)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x=x, edge_index=edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        return x

    def train_social(self, net, data, epochs, lr):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        best_accuracy = 0.0

        train_losses = []
        train_accuracies = []

        val_losses = []
        val_accuracies = []

        test_losses = []
        test_accuracies = []

        for ep in range(epochs+1):
            optimizer.zero_grad()
            out = net(data)
            loss = utils.masked_loss(predictions=out,
                                     labels=data.y,
                                     mask=data.train_mask)
            loss.backward()
            optimizer.step()
            train_losses += [loss.detach()]
            train_accuracy = utils.masked_accuracy(predictions=out,
                                                   labels=data.y,
                                                   mask=data.train_mask)
            train_accuracies += [train_accuracy]

            val_loss = utils.masked_loss(predictions=out,
                                         labels=data.y,
                                         mask=data.val_mask)
            val_losses += [val_loss.detach()]

            val_accuracy = utils.masked_accuracy(predictions=out,
                                                 labels=data.y,
                                                 mask=data.val_mask)
            val_accuracies += [val_accuracy]

            test_accuracy = utils.masked_accuracy(predictions=out,
                                                  labels=data.y,
                                                  mask=data.test_mask)
            test_accuracies += [test_accuracy]
            if np.round(val_accuracy, 4) > np.round(best_accuracy, 4):
                print("Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f}, Val_Accuracy: {:.4f}, Test_Accuracy: {:.4f}"
                      .format(ep+1, epochs, loss.item(), train_accuracy, val_accuracy,  test_accuracy))
                best_accuracy = val_accuracy

        return train_accuracies, val_accuracies, test_accuracy

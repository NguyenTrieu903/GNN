from torch_geometric.nn import GATv2Conv
from utils import accuracy
import torch.nn.functional as F
import torch
import numpy as np
from utils import batches
np.random.seed(0)


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, dim_out, learning_rate, heads=8):
        super().__init__()
        self.train_acc_all_gat = []
        self.val_acc_all_gat = []
        self.learning_rate = learning_rate

        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=heads)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer

        self.train()
        for epoch in range(epochs+1):
            # Training
            optimizer.zero_grad()
            _, out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1),
                           data.y[data.train_mask])

            loss.backward()
            optimizer.step()

            # Validation
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                               data.y[data.val_mask])

            # Print metrics every 10 epochs
            if (epoch % 10 == 0):
                self.train_acc_all_gat.append(acc)
                self.val_acc_all_gat.append(val_acc)
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                      f' {acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')
        return self.train_acc_all_gat, self.val_acc_all_gat

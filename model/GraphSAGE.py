from torch_geometric.nn import SAGEConv
from utils import accuracy
import torch.nn.functional as F
import torch
import numpy as np
from utils import batches
np.random.seed(0)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, dim_in, dim_h, dim_out, learning_rate):
        super().__init__()
        self.train_acc_all_gsea = []
        self.val_acc_all_gsea = []
        self.learning_rate = learning_rate
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer
        train_loader = batches(data)

        self.train()
        for epoch in range(epochs+1):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            # Train on batches
            for batch in train_loader:
                optimizer.zero_grad()
                _, out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask],
                                 batch.y[batch.train_mask])
                total_loss += loss
                acc += accuracy(out[batch.train_mask].argmax(dim=1),
                                batch.y[batch.train_mask])

                loss.backward()
                optimizer.step()

                # Validation
                val_loss += criterion(out[batch.val_mask],
                                      batch.y[batch.val_mask])
                val_acc += accuracy(out[batch.val_mask].argmax(dim=1),
                                    batch.y[batch.val_mask])

            # Print metrics every 10 epochs
            if (epoch % 10 == 0):
                self.train_acc_all_gsea.append(acc/len(train_loader))
                self.val_acc_all_gsea.append(val_acc/len(train_loader))
                print(f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f} '
                      f'| Train Acc: {acc/len(train_loader)*100:>6.2f}% | Val Loss: '
                      f'{val_loss/len(train_loader):.2f} | Val Acc: '
                      f'{val_acc/len(train_loader)*100:.2f}%')
        return self.train_acc_all_gsea, self.val_acc_all_gsea

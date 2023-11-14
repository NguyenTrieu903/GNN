from torch_geometric.loader import NeighborLoader
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc


def batches(data):
    return NeighborLoader(
        data,
        num_neighbors=[5, 10],
        batch_size=16,
        input_nodes=data.train_mask,
    )


def feature_distribution(data_raw):
    feats = []
    feat_counts = []
    for i in range(len(data_raw)):
        feat_counts += [len(data_raw[str(i)])]
        feats += data_raw[str(i)]
    return feats, feat_counts


def encode_data(data_raw, light=False, n=60):
    feats, feat_counts = feature_distribution(data_raw)

    if light == True:
        nodes_included = n
    elif light == False:
        nodes_included = len(data_raw)

    data_encoded = {}
    for i in range(nodes_included):
        one_hot_feat = np.array([0]*(max(feats)+1))
        this_feat = data_raw[str(i)]
        one_hot_feat[this_feat] = 1
        data_encoded[str(i)] = list(one_hot_feat)

    if light == True:
        sparse_feat_matrix = np.zeros((1, max(feats)+1))
        for j in range(nodes_included):
            temp = np.array(data_encoded[str(j)]).reshape(1, -1)
            sparse_feat_matrix = np.concatenate(
                (sparse_feat_matrix, temp), axis=0)
        sparse_feat_matrix = sparse_feat_matrix[1:, :]
        return (data_encoded, sparse_feat_matrix)
    elif light == False:
        return (data_encoded, None)


def construct_graph(target_data, edge_data, data_encoded, light=False):
    node_features_list = list(data_encoded.values())
    node_features = torch.tensor(node_features_list)
    node_labels = torch.tensor(target_data['mature'].values)
    edges_list = edge_data.values.tolist()
    edge_index01 = torch.tensor(edges_list, dtype=torch.long).T
    edge_index02 = torch.zeros(edge_index01.shape, dtype=torch.long)  # .T
    edge_index02[0, :] = edge_index01[1, :]
    edge_index02[1, :] = edge_index01[0, :]
    edge_index0 = torch.cat((edge_index01, edge_index02), axis=1)
    g = Data(x=node_features, y=node_labels, edge_index=edge_index0)
    g_light = Data(x=node_features[:, 0:2],
                   y=node_labels,
                   edge_index=edge_index0[:, :55])
    if light:
        return (g_light)
    else:
        return (g)


def masked_loss(predictions, labels, mask):
    criterion = nn.CrossEntropyLoss()
    labels = labels.long()
    mask = mask.float()
    mask = mask/torch.mean(mask)
    loss = criterion(predictions, labels)
    loss = loss*mask
    loss = torch.mean(loss)
    return (loss)


def masked_accuracy(predictions, labels, mask):
    mask = mask.float()
    mask /= torch.mean(mask)
    accuracy = (torch.argmax(predictions, axis=1) == labels).long()
    accuracy = mask*accuracy
    accuracy = torch.mean(accuracy)
    return (accuracy)


def train(net, data, epochs, lr):
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
        loss = masked_loss(predictions=out,
                           labels=data.y,
                           mask=data.train_mask)
        loss.backward()
        optimizer.step()
        train_losses += [loss.detach()]
        train_accuracy = masked_accuracy(predictions=out,
                                         labels=data.y,
                                         mask=data.train_mask)
        train_accuracies += [train_accuracy]

        val_loss = masked_loss(predictions=out,
                               labels=data.y,
                               mask=data.val_mask)
        val_losses += [val_loss.detach()]

        val_accuracy = masked_accuracy(predictions=out,
                                       labels=data.y,
                                       mask=data.val_mask)
        val_accuracies += [val_accuracy]

        test_accuracy = masked_accuracy(predictions=out,
                                        labels=data.y,
                                        mask=data.test_mask)
        test_accuracies += [test_accuracy]
        if np.round(val_accuracy, 4) > np.round(best_accuracy, 4):
            print("Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f}, Val_Accuracy: {:.4f}, Test_Accuracy: {:.4f}"
                  .format(ep+1, epochs, loss.item(), train_accuracy, val_accuracy,  test_accuracy))
            best_accuracy = val_accuracy

    return train_accuracies, val_accuracies, test_accuracy

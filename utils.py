from torch_geometric.loader import NeighborLoader


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

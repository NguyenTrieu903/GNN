from torch_geometric.datasets import Planetoid
# Import dataset from PyTorch Geometric


def get_data(data_name):
    data = Planetoid(root=".", name=data_name)
    return data


def get_Information_data(data_name):
    dataset = get_data(data_name)[0]
    # Print information about the dataset
    print(f'Dataset: {dataset}')
    print('-------------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Print information about the graph
    print(f'\nGraph:')
    print('------')
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')
    # return data

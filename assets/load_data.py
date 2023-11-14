from torch_geometric.datasets import Planetoid, Twitch, GitHub
# Import dataset from PyTorch Geometric


def get_data(data_name):
    data = Planetoid(root=".", name=data_name)
    return data


def get_data_twitch(data_name):
    data = Twitch(root=".", name=data_name)
    return data


def get_data_GitHub():
    data = GitHub(root=".")
    return data

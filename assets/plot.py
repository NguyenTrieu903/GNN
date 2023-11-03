import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.utils import degree
from collections import Counter
import numpy as np


def plotData(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(18, 18))
    plt.axis('off')
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, seed=0),
                     with_labels=False,
                     node_size=50,
                     node_color=data.y,
                     width=2,
                     edge_color="grey"
                     )
    plt.show()


def plot_node_degres(data):
    degrees = degree(data.edge_index[0]).numpy()

    # Count the number of nodes for each degree
    numbers = Counter(degrees)

    # Bar plot
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlabel('Node degree')
    ax.set_ylabel('Number of nodes')
    plt.bar(numbers.keys(),
            numbers.values(),
            color='#0A047A')
    


def plot_model(train_acc, val_acc):
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(train_acc) + 1),
             train_acc, label='train', c='blue')
    plt.plot(np.arange(1, len(val_acc) + 1),
             val_acc, label='validation', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accurarcy')
    plt.title('Compare accurancy')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.savefig('gat_gcn_loss.png')
    plt.show()

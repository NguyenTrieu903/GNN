import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.utils import degree
import networkx as nx
from torch_geometric.utils import to_networkx
import seaborn as sns
import numpy as np
from stellargraph import datasets
import collections
from assets import functions_data
import utils

st.set_page_config(page_title="Plot", page_icon="📉")

col1, col2 = st.columns(2)
app_mode = st.session_state['app_mode']


def add_missing_keys(counter, classes):
    for x in classes:
        if x not in counter.keys():
            counter[x] = 0
    return counter


def scaling(array):
    return array / sum(array)


if app_mode == 'Paper':
    user_select_value = st.session_state['dataset']
    data_str = str(user_select_value)
    data_str = data_str.replace(")", "").replace("(", "")

    data_degrees = user_select_value[0]

    if data_str == 'Cora':
        dataset = datasets.Cora()
    elif data_str == 'CiteSeer':
        dataset = datasets.CiteSeer()
    else:
        dataset = datasets.PubMedDiabetes()

    with col1:
        G, node_subjects = dataset.load()
        data = node_subjects.value_counts()
        df = data.reset_index()
        df.columns = ['subject', 'count']

        sns.set_style("whitegrid")
        if data_str == 'Cora':
            fig = plt.figure(figsize=(15, 14.5))
        elif data_str == 'CiteSeer':
            fig = plt.figure(figsize=(15, 13))
        else:
            fig = plt.figure(figsize=(15, 12.5))
        sns.barplot(data=df, x='count', y='subject', orient='h')
        plt.title('Count of Subjects')
        plt.xlabel('Count')
        plt.ylabel('Subject')
        col1.pyplot(fig)

    with col2:
        labels = data_degrees.y.numpy()
        connected_labels_set = list(
            map(lambda x: labels[x], data_degrees.edge_index.numpy()))
        connected_labels_set = np.array(connected_labels_set)

        label_connection_counts = []
        numclasses = user_select_value.num_classes
        for i in range(numclasses):
            connected_labels = connected_labels_set[:, np.where(
                connected_labels_set[0] == i)[0]]
            counter = collections.Counter(connected_labels[1])
            counter = dict(counter)
            counter = add_missing_keys(counter, range(numclasses))
            items = sorted(counter.items())
            items = [x[1] for x in items]
            label_connection_counts.append(items)
        label_connection_counts = np.array(label_connection_counts)
        # label_connection_counts.diagonal().sum() / label_connection_counts.sum()
        label_connection_counts_scaled = np.apply_along_axis(
            scaling, 1, label_connection_counts)

        fig = plt.figure(figsize=(15, 10))
        plt.rcParams["font.size"] = 13
        hm = sns.heatmap(
            label_connection_counts_scaled,
            annot=True,
            cmap='hot_r',
            fmt="1.2f",
            cbar=True,
            square=True)
        plt.xlabel("class", size=20)
        plt.ylabel("class", size=20)
        plt.tight_layout()
        plt.show()
        col2.pyplot(fig)

    degrees = degree(data_degrees.edge_index[0]).numpy()
    numbers = Counter(degrees)

    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 6))
    plt.xlabel('Node degree')
    plt.ylabel('Number of nodes')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.bar(numbers.keys(),
            numbers.values(),
            color='#047a0e')
    st.pyplot(fig)

elif app_mode == 'Twitch':
    target_data = st.session_state['target_data']
    edge_data = st.session_state['edge_data']
    data_raw = st.session_state['data_raw']

    data_degrees = functions_data.create_graph(
        data_raw, edge_data, target_data)

    with col1:
        data = target_data['mature'].value_counts()
        df = data.reset_index()
        df.columns = ['subject', 'count']

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(15, 12.5))
        sns.barplot(data=df, x='count', y='subject', orient='h')
        plt.title('Count of Subjects')
        plt.xlabel('Count')
        plt.ylabel('Subject')
        col1.pyplot(fig)

    with col2:
        labels = data_degrees.y.numpy()
        connected_labels_set = list(
            map(lambda x: labels[x], data_degrees.edge_index.numpy()))
        connected_labels_set = np.array(connected_labels_set)

        label_connection_counts = []
        numclasses = 2
        for i in range(numclasses):
            connected_labels = connected_labels_set[:, np.where(
                connected_labels_set[0] == i)[0]]
            counter = collections.Counter(connected_labels[1])
            counter = dict(counter)
            counter = add_missing_keys(counter, range(numclasses))
            items = sorted(counter.items())
            items = [x[1] for x in items]
            label_connection_counts.append(items)
        label_connection_counts = np.array(label_connection_counts)
        # label_connection_counts.diagonal().sum() / label_connection_counts.sum()
        label_connection_counts_scaled = np.apply_along_axis(
            scaling, 1, label_connection_counts)

        fig = plt.figure(figsize=(15, 10))
        plt.rcParams["font.size"] = 13
        hm = sns.heatmap(
            label_connection_counts_scaled,
            annot=True,
            cmap='hot_r',
            fmt="1.2f",
            cbar=True,
            square=True)
        plt.xlabel("class", size=20)
        plt.ylabel("class", size=20)
        plt.tight_layout()
        plt.show()
        col2.pyplot(fig)

    feats, feat_counts = utils.feature_distribution(data_raw)

    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 6))
    plt.hist(feat_counts, bins=20)
    plt.title("Number of features per graph distribution")
    st.pyplot(fig)

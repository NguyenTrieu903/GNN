import streamlit as st
from assets import load_data
import model
import utils
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from collections import Counter
from PIL import Image
import json
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit as masking
import torch
import torch.nn as nn

st.set_page_config(page_title="Data", page_icon="🗃️")
app_mode = st.sidebar.selectbox(
    '🗃️ Data', ['Paper', 'Twitch'])

# page_bg_img = '''
#     <style>
#     .stApp {
#         background: url("https://miro.medium.com/v2/resize:fit:1029/1*txzoFgR0XvAy4PpIgbUyvQ.png");
#         background-size: cover;
#     }
#     </style>
# '''
# st.markdown(page_bg_img, unsafe_allow_html=True)
col1, col2 = st.columns((3, 7))
flag_twitch = False
# for key in st.session_state.keys():
#     del st.session_state[key]
plot_data = ""
num_features = 0
num_classes = 0

if app_mode == 'Paper':
    ques = st.sidebar.radio(
        "Paper", ['Cora', 'CiteSeer', 'PubMed'])

    if ques == 'CiteSeer':
        data = load_data.get_data("CiteSeer")
        dataset = data[0]
        plot_data = dataset

        st.session_state.dataset = data

        data_str = str(data)
        data_str = data_str.replace(")", "").replace("(", "")

        col1.write(f'Dataset: {data_str}')
        col1.write('-------------------')
        col1.write(f'Number of graphs: {len(data)}')
        col1.write(f'Number of nodes: {dataset.x.shape[0]}')
        col1.write(f'Number of features: {data.num_features}')
        num_features = data.num_features
        col1.write(f'Number of classes: {data.num_classes}')
        num_classes = data.num_classes

        # Print information about the graph
        col1.write(f'\nGraph:')
        col1.write('------')
        col1.write(f'Edges are directed: {dataset.is_directed()}')
        col1.write(f'Graph has isolated nodes: {dataset.has_isolated_nodes()}')
        col1.write(f'Graph has loops: {dataset.has_self_loops()}')

        image = Image.open('./assets/img/CiteSeer.jpg')
        col2.image(image, caption='Top 10 nodes with the highest degree')
        col2.write("The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.")

    elif ques == 'Cora':
        data = load_data.get_data("Cora")
        dataset = data[0]
        plot_data = dataset

        # if 'dataset' not in st.session_state:
        st.session_state.dataset = data

        data_str = str(data)
        data_str = data_str.replace(")", "").replace("(", "")

        col1.write(f'Dataset: {data_str}')
        col1.write('-------------------')
        col1.write(f'Number of graphs: {len(data)}')
        col1.write(f'Number of nodes: {dataset.x.shape[0]}')
        col1.write(f'Number of features: {data.num_features}')
        col1.write(f'Number of classes: {data.num_classes}')

        # Print information about the graph
        col1.write(f'\nGraph:')
        col1.write('------')
        col1.write(f'Edges are directed: {dataset.is_directed()}')
        col1.write(f'Graph has isolated nodes: {dataset.has_isolated_nodes()}')
        col1.write(f'Graph has loops: {dataset.has_self_loops()}')

        image = Image.open('./assets/img/cora.jpg')
        col2.image(image, caption='Top 10 nodes with the highest degree')
        col2.write("The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.")

    elif ques == 'PubMed':
        data = load_data.get_data("PubMed")
        dataset = data[0]
        plot_data = dataset

        # if 'dataset' not in st.session_state:
        st.session_state.dataset = data

        data_str = str(data)
        data_str = data_str.replace(")", "").replace("(", "")

        col1.write(f'Dataset: {data_str}')
        col1.write('-------------------')
        col1.write(f'Number of graphs: {len(data)}')
        col1.write(f'Number of nodes: {dataset.x.shape[0]}')
        col1.write(f'Number of features: {data.num_features}')
        col1.write(f'Number of classes: {data.num_classes}')

        # Print information about the graph
        col1.write(f'\nGraph:')
        col1.write('------')
        col1.write(f'Edges are directed: {dataset.is_directed()}')
        col1.write(f'Graph has isolated nodes: {dataset.has_isolated_nodes()}')
        col1.write(f'Graph has loops: {dataset.has_self_loops()}')

        image = Image.open('./assets/img/PubMed.jpg')
        col2.image(image, caption='Top 10 nodes with the highest degree')
        col2.write("The Pubmed dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.")

if app_mode == 'Twitch':
    flag_twitch = True
    if 'twitch' not in st.session_state:
        st.session_state.twitch = 'twitch'

    ques = st.sidebar.radio(
        "Twitch", ['DE', 'ENGB', 'ES', 'PTBR', 'RU'])
    if ques == 'DE':
        with open("./data/twitch/DE/DE.json") as json_data:
            data_raw = json.load(json_data)

        edge_data = pd.read_csv('./data/twitch/DE/DE_edges.csv')
        target_data = pd.read_csv('./data/twitch/DE/DE_target.csv')
        target_data['mature'] = target_data['mature'].astype(int)

        data = load_data.get_data_twitch(ques)
        dataset = data[0]

        data_str = str(data)
        data_str = data_str.replace(")", "").replace("(", "")
        data_str = data_str + " " + ques

        col1.write(f'Dataset: {data_str}')
        col1.write('-------------------')
        col1.write(f'Number of graphs: {len(data)}')
        col1.write(f'Number of nodes: {dataset.x.shape[0]}')
        col1.write(f'Number of features: {data.num_features}')
        col1.write(f'Number of classes: {data.num_classes}')

        # Print information about the graph
        col1.write(f'\nGraph:')
        col1.write('------')
        col1.write(f'Edges are directed: {dataset.is_directed()}')
        col1.write(f'Graph has isolated nodes: {dataset.has_isolated_nodes()}')
        col1.write(f'Graph has loops: {dataset.has_self_loops()}')

        st.session_state.data_raw = data_raw
        st.session_state.edge_data = edge_data
        st.session_state.target_data = target_data

        col2.write(edge_data.head())
        col2.write(target_data.head())
        col2.write("These datasets used for node classification and transfer learning are Twitch user-user networks of gamers who stream uses mature content. Nodes are the users themselves and the links are mutual friendships between them. Vertex features are extracted based on the games played and liked, location and streaming habits. Datasets share the same set of node features, this makes transfer learning across networks possible. These social networks were collected in May 2018.")

    elif ques == 'ENGB':
        with open("./data/twitch/ENGB/ENGB_features.json") as json_data:
            data_raw = json.load(json_data)

        edge_data = pd.read_csv('./data/twitch/ENGB/ENGB_edges.csv')
        target_data = pd.read_csv('./data/twitch/ENGB/ENGB_target.csv')
        target_data['mature'] = target_data['mature'].astype(int)

        data = load_data.get_data_twitch("EN")
        dataset = data[0]

        data_str = str(data)
        data_str = data_str.replace(")", "").replace("(", "")
        data_str = data_str + " " + ques

        col1.write(f'Dataset: {data_str}')
        col1.write('-------------------')
        col1.write(f'Number of graphs: {len(data)}')
        col1.write(f'Number of nodes: {dataset.x.shape[0]}')
        col1.write(f'Number of features: {data.num_features}')
        col1.write(f'Number of classes: {data.num_classes}')

        # Print information about the graph
        col1.write(f'\nGraph:')
        col1.write('------')
        col1.write(f'Edges are directed: {dataset.is_directed()}')
        col1.write(f'Graph has isolated nodes: {dataset.has_isolated_nodes()}')
        col1.write(f'Graph has loops: {dataset.has_self_loops()}')

        st.session_state.data_raw = data_raw
        st.session_state.edge_data = edge_data
        st.session_state.target_data = target_data

        col2.write(edge_data.head())
        col2.write(target_data.head())
        col2.write("These datasets used for node classification and transfer learning are Twitch user-user networks of gamers who stream uses mature content. Nodes are the users themselves and the links are mutual friendships between them. Vertex features are extracted based on the games played and liked, location and streaming habits. Datasets share the same set of node features, this makes transfer learning across networks possible. These social networks were collected in May 2018.")

    elif ques == 'ES':
        with open("./data/twitch/ES/ES_features.json") as json_data:
            data_raw = json.load(json_data)

        data = load_data.get_data_twitch(ques)
        dataset = data[0]

        data_str = str(data)
        data_str = data_str.replace(")", "").replace("(", "")
        data_str = data_str + " " + ques

        col1.write(f'Dataset: {data_str}')
        col1.write('-------------------')
        col1.write(f'Number of graphs: {len(data)}')
        col1.write(f'Number of nodes: {dataset.x.shape[0]}')
        col1.write(f'Number of features: {data.num_features}')
        col1.write(f'Number of classes: {data.num_classes}')

        # Print information about the graph
        col1.write(f'\nGraph:')
        col1.write('------')
        col1.write(f'Edges are directed: {dataset.is_directed()}')
        col1.write(f'Graph has isolated nodes: {dataset.has_isolated_nodes()}')
        col1.write(f'Graph has loops: {dataset.has_self_loops()}')

        edge_data = pd.read_csv('./data/twitch/ES/ES_edges.csv')
        target_data = pd.read_csv('./data/twitch/ES/ES_target.csv')
        target_data['mature'] = target_data['mature'].astype(int)

        st.session_state.data_raw = data_raw
        st.session_state.edge_data = edge_data
        st.session_state.target_data = target_data

        col2.write(edge_data.head())
        col2.write(target_data.head())
        col2.write("These datasets used for node classification and transfer learning are Twitch user-user networks of gamers who stream uses mature content. Nodes are the users themselves and the links are mutual friendships between them. Vertex features are extracted based on the games played and liked, location and streaming habits. Datasets share the same set of node features, this makes transfer learning across networks possible. These social networks were collected in May 2018.")

    elif ques == 'PTBR':
        with open("./data/twitch/PTBR/PTBR_features.json") as json_data:
            data_raw = json.load(json_data)

        data = load_data.get_data_twitch("PT")
        dataset = data[0]

        data_str = str(data)
        data_str = data_str.replace(")", "").replace("(", "")
        data_str = data_str + " " + ques

        col1.write(f'Dataset: {data_str}')
        col1.write('-------------------')
        col1.write(f'Number of graphs: {len(data)}')
        col1.write(f'Number of nodes: {dataset.x.shape[0]}')
        col1.write(f'Number of features: {data.num_features}')
        col1.write(f'Number of classes: {data.num_classes}')

        # Print information about the graph
        col1.write(f'\nGraph:')
        col1.write('------')
        col1.write(f'Edges are directed: {dataset.is_directed()}')
        col1.write(f'Graph has isolated nodes: {dataset.has_isolated_nodes()}')
        col1.write(f'Graph has loops: {dataset.has_self_loops()}')

        edge_data = pd.read_csv('./data/twitch/PTBR/PTBR_edges.csv')
        target_data = pd.read_csv('./data/twitch/PTBR/PTBR_target.csv')
        target_data['mature'] = target_data['mature'].astype(int)

        st.session_state.data_raw = data_raw
        st.session_state.edge_data = edge_data
        st.session_state.target_data = target_data

        col2.write(edge_data.head())
        col2.write(target_data.head())
        col2.write("These datasets used for node classification and transfer learning are Twitch user-user networks of gamers who stream uses mature content. Nodes are the users themselves and the links are mutual friendships between them. Vertex features are extracted based on the games played and liked, location and streaming habits. Datasets share the same set of node features, this makes transfer learning across networks possible. These social networks were collected in May 2018.")

    elif ques == 'RU':
        with open("./data/twitch/RU/RU_features.json") as json_data:
            data_raw = json.load(json_data)

        data = load_data.get_data_twitch(ques)
        dataset = data[0]

        data_str = str(data)
        data_str = data_str.replace(")", "").replace("(", "")
        data_str = data_str + " " + ques

        col1.write(f'Dataset: {data_str}')
        col1.write('-------------------')
        col1.write(f'Number of graphs: {len(data)}')
        col1.write(f'Number of nodes: {dataset.x.shape[0]}')
        col1.write(f'Number of features: {data.num_features}')
        col1.write(f'Number of classes: {data.num_classes}')

        # Print information about the graph
        col1.write(f'\nGraph:')
        col1.write('------')
        col1.write(f'Edges are directed: {dataset.is_directed()}')
        col1.write(f'Graph has isolated nodes: {dataset.has_isolated_nodes()}')
        col1.write(f'Graph has loops: {dataset.has_self_loops()}')

        edge_data = pd.read_csv('./data/twitch/RU/RU_edges.csv')
        target_data = pd.read_csv('./data/twitch/RU/RU_target.csv')
        target_data['mature'] = target_data['mature'].astype(int)

        st.session_state.data_raw = data_raw
        st.session_state.edge_data = edge_data
        st.session_state.target_data = target_data

        col2.write(edge_data.head())
        col2.write(target_data.head())
        col2.write("These datasets used for node classification and transfer learning are Twitch user-user networks of gamers who stream uses mature content. Nodes are the users themselves and the links are mutual friendships between them. Vertex features are extracted based on the games played and liked, location and streaming habits. Datasets share the same set of node features, this makes transfer learning across networks possible. These social networks were collected in May 2018.")

    st.session_state.que = ques

st.session_state.app_mode = app_mode

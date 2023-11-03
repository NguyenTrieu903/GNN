import streamlit as st
from assets import load_data, plot
import model
import utils
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from collections import Counter
from PIL import Image

# from sessionstate import SessionState

st.set_page_config(page_title="Data", page_icon="🗃️")
app_mode = st.sidebar.selectbox(
    '🗃️ Data', ['CiteSeer', 'PubMed', 'Cora'])

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

for key in st.session_state.keys():
    del st.session_state[key]

# state = SessionState.get(selected_item=None)
plot_data = ""
if app_mode == 'CiteSeer':
    data = load_data.get_data("CiteSeer")
    dataset = data[0]
    plot_data = dataset

    if 'dataset' not in st.session_state:
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

    image = Image.open('./assets/img/CiteSeer.jpg')
    col2.image(image, caption='Top 10 nodes with the highest degree')
    col2.write("The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.")


if app_mode == 'Cora':
    data = load_data.get_data("Cora")
    dataset = data[0]
    plot_data = dataset

    if 'dataset' not in st.session_state:
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


if app_mode == 'PubMed':
    data = load_data.get_data("PubMed")
    dataset = data[0]
    plot_data = dataset

    if 'dataset' not in st.session_state:
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

    image = Image.open('./assets/img/Pubmed.jpg')
    col2.image(image, caption='Top 10 nodes with the highest degree')
    col2.write("The Pubmed dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.")

# degrees = degree(plot_data.edge_index[0]).numpy()

# # Count the number of nodes for each degree
# numbers = Counter(degrees)

# # Bar plot
# fig, ax = plt.subplots()
# fig = plt.figure(figsize=(5, 8))
# plt.xlabel('Epochs')
# plt.ylabel('Accurarcy')
# plt.title('Node degrees')
# plt.legend(loc='lower right', fontsize='x-large')
# plt.bar(numbers.keys(),
#         numbers.values(),
#         color='#047a0e')
# col3.plotly_chart(fig)

import streamlit as st
from model.GAT import *
from model.GraphSAGE import *
from model.GCN import *
import matplotlib.pyplot as plt
import utils
from torch_geometric.transforms import RandomNodeSplit as masking
from assets import functions_data

st.set_page_config(page_title="Model", page_icon="🛠️")

# page_bg_img = '''
#     <style>
#     .stApp {
#         background: url("https://cutewallpaper.org/21x/2olaxzatm/25-Awesome-Web-Background-Animation-Effects-Css-animation-.jpg");
#         background-size: cover;
#     }
#     </style>
# '''
# st.markdown(page_bg_img, unsafe_allow_html=True)

train_accuracies = []
val_accuracies = []
test_accuracy = 0
data = ""

with st.sidebar.container():
    with st.form(key="model"):
        hidden_channels = st.number_input(
            'Hidden channels', step=1, key="hidden_channels", value=64)

        learning_rate = st.number_input(
            'Learning rate', key="learning_rate", value=0.01)

        epochs = st.number_input(
            'Epochs', key="Epochs", value=100)
        app_mode = st.session_state['app_mode']

        if app_mode == 'Twitch' or app_mode == 'Github':
            train_split = st.number_input(
                'train_split', key="train_split", value=0.2)
            test_split = st.number_input(
                'test_split', key="test_split", value=0.6)
            val_split = st.number_input(
                'val_split', key="val_split", value=0.2)

        ques = st.radio(
            "Model", ['GCN', 'GraphSEA', 'GAT'])
        submitted = st.form_submit_button("train", type="primary")

        if ques == 'GCN':
            if app_mode == 'Twitch':
                data_raw = st.session_state['data_raw']
                edge_data = st.session_state['edge_data']
                target_data = st.session_state['target_data']
                st.write(st.session_state['que'])

                g = functions_data.create_graph(
                    data_raw, edge_data, target_data, train_split, val_split, test_split)

                if submitted:
                    criterion = nn.CrossEntropyLoss()
                    train_accuracies, val_accuracies, test_accuracy = functions_data.submit(ques,
                                                                                            g, epochs, learning_rate, g.num_edge_features, hidden_channels, 2)

            elif app_mode == 'Paper':
                user_select_value = st.session_state['dataset']
                data = user_select_value
                st.write(data)
                if submitted:
                    train_accuracies, val_accuracies, test_accuracy = functions_data.submit(ques,
                                                                                            data, epochs, learning_rate, user_select_value.num_features, hidden_channels, user_select_value.num_classes)
        if ques == 'GraphSEA':
            if app_mode == 'Twitch':
                data_raw = st.session_state['data_raw']
                edge_data = st.session_state['edge_data']
                target_data = st.session_state['target_data']

                g = functions_data.create_graph(
                    data_raw, edge_data, target_data, train_split, val_split, test_split)

                if submitted:
                    train_accuracies, val_accuracies, test_accuracy = functions_data.submit(ques,
                                                                                            g, epochs, learning_rate, g.num_edge_features, hidden_channels, 2)

            if app_mode == 'Paper':
                user_select_value = st.session_state['dataset']
                data = user_select_value
                st.write(data)
                train_accuracies, val_accuracies, test_accuracy = functions_data.submit(ques,
                                                                                        data, epochs, learning_rate, user_select_value.num_features, hidden_channels, user_select_value.num_classes)

        if ques == 'GAT':
            if app_mode == 'Twitch':
                data_raw = st.session_state['data_raw']
                edge_data = st.session_state['edge_data']
                target_data = st.session_state['target_data']

                g = functions_data.create_graph(
                    data_raw, edge_data, target_data, train_split, val_split, test_split)

                if submitted:
                    train_accuracies, val_accuracies, test_accuracy = functions_data.submit(ques,
                                                                                            g, epochs, learning_rate, g.num_edge_features, hidden_channels, 2)

            elif app_mode == 'Paper':
                user_select_value = st.session_state['dataset']
                data = user_select_value
                st.write(data)
                if submitted:
                    train_accuracies, val_accuracies, test_accuracy = functions_data.submit(ques,
                                                                                            data, epochs, learning_rate, user_select_value.num_features, hidden_channels, user_select_value.num_classes)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(12, 8))
plt.plot(np.arange(1, len(train_accuracies) + 1),
         train_accuracies, label='train', c='blue')
plt.plot(np.arange(1, len(val_accuracies) + 1),
         val_accuracies, label='validation', c='red')
plt.xlabel('epoch')
plt.ylabel('accurancy')
plt.title('Training and validation accurancy')
plt.legend(loc='lower right', fontsize='x-large')
plt.show()
st.pyplot(fig)
st.write(f'\n{ques} test accuracy: {test_accuracy*100:.2f}%\n')

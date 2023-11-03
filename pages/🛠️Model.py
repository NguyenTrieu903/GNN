import streamlit as st
from model.GCN import *
from model.GAT import *
from model.GraphSAGE import *
import matplotlib.pyplot as plt
import utils

# import SessionState
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

train_acc = []
val_acc = []
test_acc = 0
data = ""
# state = SessionState.get()
with st.sidebar.container():
    with st.form(key="model"):
        hidden_channels = st.number_input(
            'Hidden channels', step=1, key="hidden_channels", value=64)

        learning_rate = st.number_input(
            'Learning rate', key="learning_rate", value=0.01)

        epochs = st.number_input(
            'Epochs', key="Epochs", value=100)

        ques = st.radio(
            "Model", ['GCN', 'GraphSEA', 'GAT'])
        submitted = st.form_submit_button("train", type="primary")

        if ques == 'GCN':
            user_select_value = st.session_state['dataset']
            data = user_select_value
            if submitted:
                gcn = GCN(user_select_value.num_features,
                          hidden_channels, user_select_value.num_classes, learning_rate)
                train_acc, val_acc = gcn.fit(user_select_value[0], epochs)
                test_acc = utils.test(gcn, user_select_value[0])

        if ques == 'GraphSEA':
            user_select_value = st.session_state['dataset']
            data = user_select_value
            if submitted:
                graphSEA = GraphSAGE(user_select_value.num_features,
                                     hidden_channels, user_select_value.num_classes, learning_rate)
                train_acc, val_acc = graphSEA.fit(user_select_value[0], epochs)
                test_acc = utils.test(graphSEA, user_select_value[0])

        if ques == 'GAT':
            user_select_value = st.session_state['dataset']
            data = user_select_value
            if submitted:
                gat = GAT(user_select_value.num_features,
                          hidden_channels, user_select_value.num_classes, learning_rate)
                train_acc, val_acc = gat.fit(user_select_value[0], epochs)
                test_acc = utils.test(gat, user_select_value[0])


fig, ax = plt.subplots()
fig = plt.figure(figsize=(12, 8))
plt.plot(np.arange(1, len(train_acc) + 1),
         train_acc, label='train', c='blue')
plt.plot(np.arange(1, len(val_acc) + 1),
         val_acc, label='validation', c='red')
plt.xlabel('epoch')
plt.ylabel('accurancy')
plt.title('Training and validation accurancy')
plt.legend(loc='lower right', fontsize='x-large')
plt.show()
st.pyplot(fig)
st.write(f'\n{ques} test accuracy: {test_acc*100:.2f}%\n')

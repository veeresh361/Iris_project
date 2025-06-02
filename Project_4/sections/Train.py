import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from model.VeereshNetwork import VeereshNetwork
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sections.utils import (get_train_data,train_model,plot_loss_accuracy)
from config import MODEL_SAVE_PATH




OPTIMIZERS = ["SGD", "Adam", "RMSprop"]
def run():
    if 'model' not in st.session_state:
      st.session_state['model'] = None

    if 'train_data' not in st.session_state:
      st.session_state['train_data'] = None

    if 'test_data' not in st.session_state:
      st.session_state['test_data'] = None

    if 'Optimizer' not in st.session_state:
      st.session_state['Optimizer'] = None
    
    st.title("🧠 Model Training & Evaluation")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(st.session_state['train_data'].head(5))
    with st.form("ann_config_form", clear_on_submit=False):
        st.header("ANN Configuration")
        num_layers = st.number_input("Number of Hidden Layers", min_value=1, max_value=10, value=2)

        hidden_units = []
        for i in range(num_layers):
            units = st.number_input(f"Neurons in Hidden Layer {i+1}", min_value=1, max_value=1024, value=10)
            hidden_units.append(units)

        st.session_state['Optimizer'] = st.selectbox("Optimizer", OPTIMIZERS)

        submitted = st.form_submit_button("Build & Train Model")
        if submitted:
            input_dim=st.session_state['train_data'].shape[-1]
            st.session_state['model'] = VeereshNetwork(input_dim,hidden_units,output_dim=4)
            st.session_state['train_data'],st.session_state['test_data']=get_train_data(st.session_state['train_data'],
                        st.session_state['target'],
                        test_size=0.2,batch_size=16)
            
            st.session_state['model'],losses=train_model(st.session_state['model'],
                                                         epoch=10,
                                                         train_loader=st.session_state['train_data'])
            torch.save(st.session_state['model'].state_dict(), MODEL_SAVE_PATH)
            st.success('Model is trained Succesfully')
            loss_grapth=plot_loss_accuracy(losses,10)
            st.pyplot(loss_grapth)

            

            

    
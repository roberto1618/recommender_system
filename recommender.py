# Import necessary functions and scripts
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from autoencoder import autoencoder_model
from data_clean import data_clean

# Read data
data = pd.read_csv('test_data_recommender.csv', sep = ';')

# Clean data
data = data_clean(data)

# Transform data to sparse
# Add necessary variables to do that
data.quantity = 1
data.id_client_cat = data.id_client.astype('category').cat.codes
data.item_id_cat = data.item_id.astype('category').cat.codes
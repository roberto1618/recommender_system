# Import necessary functions and scripts
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from autoencoder import autoencoder_model
from data_clean import data_clean
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Autoencoder builder
def autoencoder_model(x_train, x_test, epochs = 800, batch_size = 16, encoding_dim = 5, hidden_activation = 'relu', optimizer = 'adam', loss = 'binary_crossentropy'):
    input_dim = x_train.shape[1]
    input_layer = Input(shape = (input_dim,))
    encoding_layer = Dense(encoding_dim, activation = hidden_activation)
    encoded = encoding_layer(input_layer)

    decoding_layer = Dense(x_train.shape[1], activation = 'softmax')
    decoded = decoding_layer(encoded)

    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer = optimizer, loss = loss)

    autoencoder.fit(x_train, x_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test, x_test), verbose = 1)

    return autoencoder

# Read data
data = pd.read_csv('test_data_recommender.csv', sep = ';')

# Clean data
data = data_clean(data)

# Transform data to sparse
# Add necessary variables to do that
data['quantity'] = 1
data['id_client_cat'] = data['id_client'].astype('category').cat.codes
data['item_id_cat'] = data['item'].astype('category').cat.codes

## Create sparse matrix
# Get the clients and items list
clients = list(np.sort(data['id_client_cat'].unique()))
items = list(np.sort(data['item_id_cat'].unique()))
quantity = list(data['quantity'])

# Get the rows and columns for our new matrix
rows = data['id_client_cat'].astype(float)
cols = data['item_id_cat'].astype(float)

# Contruct a sparse matrix
data_sparse = csr_matrix((quantity, (rows, cols)), shape=(len(clients), len(items)))

# Return to dense matrix
dense_matrix = np.array(data_sparse.todense())

# Divide into train and test
x_train = dense_matrix[:-4]
x_test = dense_matrix[-4:]
#x_train, x_test = train_test_split(dense_matrix, test_size = 0.2)

# Build the model
autoencoder = autoencoder_model(x_train, x_test)

# Get the prediction
predicted_items = autoencoder.predict(x_test)
predictions_df = pd.DataFrame(predicted_items)

# Add the client
predictions_df['client'] = predictions_df.index

# Build the melted matrix with recommendations
items = list(range(0, 10))
predictions_melted = pd.melt(predictions_df, id_vars = ['client'], value_vars = items)
predictions_melted["rank"] = predictions_melted.groupby("client")["value"].rank("dense", ascending=False)

# Final recommendations
top_recommendations = predictions_melted[(predictions_melted['rank']>=1) & (predictions_melted['rank']<=5)].sort_values('client')

#top_recommendations = pd.merge(top_recommendations, data, left_on = 'variable', right_on = 'item_id_cat', how = 'inner')
print(top_recommendations)
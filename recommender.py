# Import necessary functions and scripts
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
#from autoencoder import autoencoder_model
from data_clean import data_clean
from scipy.sparse import csr_matrix
#from sklearn.model_selection import train_test_split

# Initial functions
# Autoencoder builder
def autoencoder_model(x_train, epochs = 800, batch_size = 16, encoding_dim = 5, hidden_activation = 'relu', optimizer = 'adam', loss = 'binary_crossentropy'): #x_test,
    input_dim = x_train.shape[1]
    input_layer = Input(shape = (input_dim,))
    encoding_layer = Dense(encoding_dim, activation = hidden_activation)
    encoded = encoding_layer(input_layer)

    decoding_layer = Dense(x_train.shape[1], activation = 'softmax')
    decoded = decoding_layer(encoded)

    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer = optimizer, loss = loss)

    autoencoder.fit(x_train, x_train, epochs = epochs, batch_size = batch_size, verbose = 1) #, validation_data = (x_test, x_test)

    return autoencoder

# Initial parameters
n_recom = 5
final_recom = 3

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
x_train = dense_matrix #[:-4]
#x_test = dense_matrix[-4:]
#x_train, x_test = train_test_split(dense_matrix, test_size = 0.2)

# Build the model
autoencoder = autoencoder_model(x_train) #, x_test

# Get the prediction
predicted_items = autoencoder.predict(x_train) #x_test
predictions_df = pd.DataFrame(predicted_items)

# Add the client
predictions_df['client'] = predictions_df.index

# Build the melted matrix with recommendations
items = list(range(0, 10))
predictions_melted = pd.melt(predictions_df, id_vars = ['client'], value_vars = items)
predictions_melted["rank"] = predictions_melted.groupby("client")["value"].rank("dense", ascending=False)

# Final recommendations
top_recommendations = predictions_melted[(predictions_melted['rank']>=1) & (predictions_melted['rank']<=n_recom)].sort_values('client')

# Change the value of recommendation for the name
items_data = data[['item', 'item_id_cat']].drop_duplicates()
top_recommendations = pd.merge(top_recommendations, items_data, left_on = 'variable', right_on = 'item_id_cat', how = 'left')

# Pivot table to have recommendations by column
top_recommendations['rank'] = 'rec_' + top_recommendations['rank'].astype(int).astype(str)
top_recommendations = top_recommendations.pivot(index = 'client', columns = 'rank', values = 'item')
top_recommendations.columns.name = None
top_recommendations = top_recommendations.reset_index()

# Merge to with the data
top_recommendations = pd.merge(data, top_recommendations, left_on = 'id_client_cat', right_on = 'client', how = 'inner')

# Remove products that have been already purchased
for r in range(1, n_recom + 1):
    top_recommendations.loc[top_recommendations['item']==top_recommendations['rec_' + str(r)], 'rec_' + str(r)] = 'remove'

# Join by customer
for r in range(1, n_recom + 1):
    top_recommendations['rec_' + str(r)] = top_recommendations.groupby(['id_client'])['rec_' + str(r)].transform(lambda x : ' '.join(x))
    top_recommendations['rec_' + str(r)] = top_recommendations['rec_' + str(r)].str.split().apply(lambda x: ''.join(list(set(x))))
    top_recommendations.loc[top_recommendations['rec_' + str(r)].str.contains("remove"), 'rec_' + str(r)] = None

# Remove duplicates
top_recommendations = top_recommendations[['id_client'] + ['rec_' + str(r) for r in range(1, n_recom + 1)]]
top_recommendations = top_recommendations.drop_duplicates()

# Get only the final recommendations
top_recommendations = pd.melt(top_recommendations, id_vars = ['id_client'], value_vars = ['rec_' + str(r) for r in range(1, n_recom + 1)])
top_recommendations = top_recommendations.loc[top_recommendations['value'].notnull(),:]
top_recommendations['variable'] = top_recommendations['variable'].map(lambda x: x.lstrip('rec_')).astype(int)
top_recommendations['rank'] = top_recommendations.groupby('id_client')['variable'].rank('dense')
top_recommendations = top_recommendations.loc[top_recommendations['rank']<=final_recom,:]
top_recommendations['rank'] = 'rec_' + top_recommendations['rank'].astype(int).astype(str)
top_recommendations = top_recommendations.pivot(index = 'id_client', columns = 'rank', values = 'value')
top_recommendations.columns.name = None
top_recommendations = top_recommendations.reset_index()


print(top_recommendations)
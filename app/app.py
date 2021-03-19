# Read packages
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import base64
import datetime
import io
import dash_table
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input as _Input
from tensorflow.keras.models import Model
#from data_clean import data_clean
from scipy.sparse import csr_matrix

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

# Define tabs styles
tabs_styles = {
    'height': '44px'
}

tab_style = {
    'borderTop': '1px solid #000000',
    'borderBottom': '1px solid #000000',
    'backgroundColor': '#F69BA0',
    'fontSize': '20px',
    'color': 'white',
    'padding': '10px'
}

tab_selected_style = {
    'borderTop': '2px solid #000000',
    'borderBottom': '2px solid #000000',
    'backgroundColor': '#F93640',
    'fontSize': '30px',
    'fontWeight': 'bold',
    'color': 'white',
    'padding': '3px'
}

# Colors of the background and text
colors = {
    'background': '#FFFFFF',
    'text': '#000000'
}

# Content for uploading the data
content_reco = html.Div([
    dbc.Row(
        [
            dbc.Col(                    
                dcc.Upload(
                    id = 'upload-data',
                    children = html.Div([
                        'Upload your data here. Drag or ', html.A('Select from folder...')
                    ]),
                    style = {
                        'width': '25%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple = True
                    )
            
            ),
            dbc.Col(
                html.Div(id = 'recommendation-output')
            )
        ]
    )
])
# Layout
app.layout = html.Div(style = {'backgroundColor': colors['background']}, children = [
    html.H1(
        children = 'Recommendation System',
        style = {
            'textAlign': 'center',
            'color': colors['text'],
            'background': '#F7DCDD'
        }
    ),
    # Define tabs
    dcc.Tabs(id = "tabs-styled-with-inline", value = 'reco', children = [
        # First tab. Show the recommendations by customer
        dcc.Tab(label = 'Recommender', value = 'reco', style = tab_style, selected_style = tab_selected_style,
                children = [
                    html.H3('Upload your data and get the recommendations!'),
                    content_reco,
                ]),
        dcc.Tab(label = 'Summary items', value = 'summ', style = tab_style, selected_style = tab_selected_style,
                children = [
                    html.Div(id = 'bar-plot-data'),
                ]),
        # Third tab. More info about the app
        dcc.Tab(label = 'More info', value = 'info', style = tab_style, selected_style = tab_selected_style),
    ], style = tabs_styles),
    html.Div(id = 'tabs-content-inline'),
])

# Function to read the uploaded file
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep = ';')
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

# Function that returns a barplot for the summary
def bar_plot_summary(data):
    df = data.groupby('item', as_index = False).count()
    # Return the barplot
    return html.Div([
        dcc.Graph(
            figure = go.Figure(data = [
            go.Bar(name = df.columns.values[0], x = pd.unique(df['item']), y = df['id_client']),
            ])
            ),        
    ])

# Function to build the autoencoder
# Autoencoder builder
def autoencoder_model(x_train, epochs = 800, batch_size = 16, encoding_dim = 5, hidden_activation = 'relu', optimizer = 'adam', loss = 'binary_crossentropy'): #x_test,
    input_dim = x_train.shape[1]
    input_layer = _Input(shape = (input_dim,))
    encoding_layer = Dense(encoding_dim, activation = hidden_activation)
    encoded = encoding_layer(input_layer)

    decoding_layer = Dense(x_train.shape[1], activation = 'softmax')
    decoded = decoding_layer(encoded)

    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer = optimizer, loss = loss)

    autoencoder.fit(x_train, x_train, epochs = epochs, batch_size = batch_size, verbose = 1) #, validation_data = (x_test, x_test)

    return autoencoder

# Function for generating tables with recommendations
def generate_table(dataframe, max_rows = 30):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# Function for cleaning data
def data_clean(data):
    # Remove duplicate rows
    data_cleaned = data.drop_duplicates()
    return data_cleaned

# Function to take the data from the previous functions and return the barplot for the summary
@app.callback(Output('bar-plot-data', 'children'),
            [Input('upload-data', 'contents')],
            [State('upload-data', 'filename'),
            State('upload-data', 'last_modified')])
def update_bar_plot(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            bar_plot_summary(parse_contents(c, n, d)) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

# Function to calculate the recomendations and return the table
@app.callback(Output('recommendation-output', 'children'),
            [Input('upload-data', 'contents')],
            [State('upload-data', 'filename'),
            State ('upload-data', 'last_modified')])
def recommender_fun(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        # Initial parameters
        n_recom = 5
        final_recom = 3

        # Read data
        data = parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])

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
        top_recommendations['item'] = top_recommendations.groupby(['id_client'])['item'].transform(lambda x : ', '.join(x))
        top_recommendations = top_recommendations[['id_client', 'item'] + ['rec_' + str(r) for r in range(1, n_recom + 1)]]
        top_recommendations = top_recommendations.drop_duplicates()

        # Get only the final recommendations
        top_recommendations = pd.melt(top_recommendations, id_vars = ['id_client', 'item'], value_vars = ['rec_' + str(r) for r in range(1, n_recom + 1)])
        top_recommendations = top_recommendations.loc[top_recommendations['value'].notnull(),:]
        top_recommendations['variable'] = top_recommendations['variable'].map(lambda x: x.lstrip('rec_')).astype(int)
        top_recommendations['rank'] = top_recommendations.groupby(['id_client', 'item'])['variable'].rank('dense')
        top_recommendations = top_recommendations.loc[top_recommendations['rank']<=final_recom,:]
        top_recommendations['rank'] = 'rec_' + top_recommendations['rank'].astype(int).astype(str)
        final_recommendations = top_recommendations.pivot(index = 'id_client', columns = 'rank', values = 'value')
        final_recommendations.columns.name = None
        final_recommendations = final_recommendations.reset_index()
        final_recommendations = pd.merge(final_recommendations, top_recommendations[['id_client', 'item']].drop_duplicates(), on = 'id_client', how = 'left')
        final_recommendations = final_recommendations[['id_client', 'item'] + ['rec_' + str(r) for r in range(1, final_recom + 1)]]
        final_recommendations = final_recommendations.rename(columns = {'item': 'purchases'})

        final_recommendations = generate_table(final_recommendations)
        return final_recommendations

# Execute the app
if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1', debug=True)
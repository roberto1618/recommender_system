# Read packages
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import base64
import datetime
import io
import dash_table
import pandas as pd

app = dash.Dash()

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
                    # Upload the data
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
                    ),
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

# Execute the app
if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1', debug=True)
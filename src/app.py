import pandas as pd
import dash
import pickle
from joblib import load
from dash import dcc, html
from dash.dash import no_update
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
from opencage.geocoder import OpenCageGeocode

# Load the data
df = pd.read_csv("pos_neg_dummy_data.csv")

# Initialize the OpenCage Geocoder
geocoder = OpenCageGeocode("02d943c6618e41329313406447a26533")

# Initialize the Dash app
app = dash.Dash(
    external_stylesheets=[dbc.themes.CERULEAN]
)

# Create plots
def create_location_plot(category=None):
    filtered_df = df if category is None else df[df['category'] == category]
    fig = px.scatter_mapbox(filtered_df, lat="lat", lon="long", hover_name="merchant", hover_data=["amt", "category"],
                            color_discrete_sequence=["blue"], zoom=3, height=400, size="amt",title="Fraud across locations")
    fig.update_layout(mapbox_style="open-street-map")
    return fig
    

def create_fraud_plot(category=None):
    filtered_df = df if category is None else df[df['category'] == category]
    fraud_df = filtered_df[filtered_df['is_fraud'] == 1]
    fig = px.histogram(fraud_df, x="category", title="Fraud Transactions by Category", height=400)
    if category is not None:
        fig.data[0].marker.color = ['red' if cat == category else 'blue' for cat in fig.data[0].x]
    return fig

categories = df['category'].unique()
genders = df['gender'].unique()
with open('job_list.pkl', 'rb') as f:
    jobs = pickle.load(f)
with open('merchant_lst.pkl', 'rb') as f:
    merchants = pickle.load(f)
# App layout
app.layout = html.Div([
    html.H1("Credit Card Transactions Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label('Select a category of fraud:'),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': cat, 'value': cat} for cat in categories],
            value=None,
            placeholder="Select a category"
        ),
        html.Br(),
        html.Label('Number of frauds in selected category:'),
        dcc.Input(id='category-fraud-count', type="text", disabled=True),
    ], style={'width': '50%', 'margin': 'auto'}),
    
    html.Div([
        dcc.Graph(id="location-plot", figure=create_location_plot()),
        dcc.Graph(id="fraud-plot"),
        html.Div(id="location-info"),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'margin': 'auto', 'marginTop': '20px'}),
    
    html.Div([
        html.Div([
            html.Div([            
                html.Label('Credit Card Number', style={'float': 'left', 'width': '50%'}),
                dcc.Input(id='cc_num', type='text', value='', style={'width': '50%'}),
                
                html.Label('Merchant Name', style={'float': 'left', 'width': '50%'}),
                dcc.Dropdown(id='merchant', options=[{'label': merchant, 'value': merchant} for merchant in merchants], value=''),

                html.Label('Category', style={'float': 'left', 'width': '50%'}),
                dcc.Dropdown(id='category', options=[{'label': cat, 'value': cat} for cat in categories], value='misc_net'),
                
                html.Label('Amount', style={'float': 'left', 'width': '50%'}),
                dcc.Input(id='amt', type='number', value=10, style={'width': '50%'}),
            ], style={'width': '50%', 'float': 'left'}),
            
            html.Div([
                html.Label('Gender', style={'float': 'left', 'width': '50%'}),
                dcc.Dropdown(id='gender', options=[{'label': gender, 'value': gender} for gender in genders], value='F'),

                html.Label('City', style={'float': 'left', 'width': '50%'}),
                dcc.Input(id='city', type='text', value='', style={'width': '50%'}),
                
                html.Label('State', style={'float': 'left', 'width': '50%'}),
                dcc.Input(id='state', type='text', value='', style={'width': '50%'}),
                                
                html.Label('Job', style={'float': 'left', 'width': '50%'}),
                dcc.Dropdown(id='job', options=[{'label': job, 'value': job} for job in jobs], value='counselling'),
            ], style={'width': '50%', 'float': 'right'}),
        ], style={'width': '80%', 'margin': 'auto'}),
        
        html.Div([
            html.Button('Predict Fraud', id='predict-button', n_clicks=0, style={'margin': 'auto', 'display': 'block'}),
        ], style={'width': '50%', 'margin': 'auto', 'marginTop': '20px'}),
    ]),
    
    # Output for prediction result
    html.Div(id='prediction-output', style={'textAlign': 'center', 'marginTop': '20px'})
])

# Function to get place name from latitude and longitude
def get_place_name(lat, lon):
    results = geocoder.reverse_geocode(lat, lon)
    if results and len(results):
        return results[0]['formatted']
    else:
        return "Unknown"
# Callback to update location plot based on category selection
@app.callback(
    Output("location-plot", "figure"),
    [Input("category-dropdown", "value"),
     Input("location-plot", "relayoutData")]
)
def update_location_plot(selected_category,relayoutData):
    if relayoutData is not None:
        if 'mapbox.center' in relayoutData and 'mapbox.zoom' in relayoutData:
            center_lat = relayoutData['mapbox.center']['lat']
            center_lon = relayoutData['mapbox.center']['lon']
            zoom = relayoutData['mapbox.zoom']
            print(f"Latitude: {center_lat}, Longitude: {center_lon}, Zoom: {zoom}")
            place_name = get_place_name(center_lat, center_lon)
            title = f"Fraud across locations - Zoomed in on {place_name}"
            return create_location_plot(selected_category).update_layout(title=title,mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom))
    return create_location_plot(selected_category)


# Callback to update fraud plot based on category selection
@app.callback(
    Output("fraud-plot", "figure"),
    [Input("category-dropdown", "value")]
)
def update_fraud_plot(selected_category):
    return create_fraud_plot(selected_category)


# Callback to update count based on category selection
@app.callback(
    Output("category-fraud-count", "value"),
    [Input("category-dropdown", "value")]
)
def update_category_count(category=None):
    if category is not None:
        count = len(df[df['category'] == category])
        return f"{count}"
    else:
        return ""
    
# Display the location info
@app.callback(
    Output("location-info", "children"),
    [Input("location-plot", "clickData")]
)
def display_location_info(clickData):
    if clickData and 'points' in clickData:
        lat = clickData['points'][0]['lat']
        lon = clickData['points'][0]['lon']
        place_name = get_place_name(lat, lon)
        return f"You clicked on: {place_name}"
    return ""

# Define callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [#State('trans_date_trans_time', 'value'),
    State('category', 'value'),
    State('cc_num', 'value'),
    State('merchant', 'value'),
    State('amt', 'value'),
    State('gender', 'value'),
    State('job','value'),
    State('city', 'value'),
    State('state', 'value'),
    ])

def predict_fraud_callback(n_clicks, cc_num, merchant, category, amt, gender, city, state, job):
    if n_clicks > 0:
        transaction_data = pd.DataFrame({
            # 'trans_date_time': [trans_date_time],
            'category': [category],
            'cc_num': [cc_num],
            'merchant': [merchant],
            'amt': [amt],
            'gender': [gender],
            'job': [job],
            'city': [city],
            'state': [state],
            # 'zip': [zip_code],
            # 'lat': [lat],
            # 'long': [long],
            # 'merch_lat': [merch_lat],
            # 'merch_long': [merch_long]
        })
        prediction = predict_fraud(transaction_data)
        return prediction
    else:
        return no_update


def predict_fraud(dataframe):
    with open('pipeline.joblib', 'rb') as f:
        loaded_pipeline = load('pipeline.joblib')
    dataframe = dataframe[['merchant','category','amt','gender','job']]
    print(dataframe)
    pred = loaded_pipeline.predict(dataframe)
    print(pred)
    print('Classifier ran successfully.')
    return "Fraudulent" if pred[0] == 1 else "Not Fraudulent"
# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

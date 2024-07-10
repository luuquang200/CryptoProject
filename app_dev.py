import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table as dt
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from data_utils import DataUtils, TimeFrame
from model_helper import ModelHelper

# Load environment variables từ file .env
load_dotenv()

# Define style color
colors = {"background": "#191d21", "text": "#ffffff"}

# List of crypto pairs to display
crypto_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

external_stylesheets = [dbc.themes.SLATE]

# Adding css
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.Div(
            [  # Header Div
                dbc.Row(
                    [
                        dbc.Col(
                            html.Header(
                                [
                                    html.H1(
                                        "Cryptocurrency Dashboard",
                                        style={
                                            "textAlign": "center",
                                            "color": colors["text"],
                                        },
                                    )
                                ]
                            )
                        )
                    ]
                )
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            [  # Dropdown Div
                dbc.Row(
                    [
                        dbc.Col(  # Tickers
                            dcc.Dropdown(
                                id="crypto_pair",
                                options=[
                                    {"label": pair, "value": pair}
                                    for pair in crypto_pairs
                                ],
                                searchable=True,
                                value="BTCUSDT",
                                placeholder="Select cryptocurrency pair",
                            ),
                            width={"size": 3, "offset": 3},
                        ),
                        dbc.Col(  # Graph type
                            dcc.Dropdown(
                                id="chart",
                                options=[
                                    {"label": "line", "value": "Line"},
                                    {"label": "candlestick", "value": "Candlestick"},
                                    {"label": "line and candlestick", "value": "Line and Candlestick"},
                                    {"label": "sma", "value": "SMA"},
                                ],
                                value="Line",
                                style={"color": "#000000"},
                            ),
                            width={"size": 3},
                        ),
                        dbc.Col(  # Button
                            dbc.Button(
                                "Plot",
                                id="submit-button-state",
                                className="mr-1",
                                n_clicks=0,
                            ),
                            width={"size": 2},
                        ),
                    ]
                )
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="live_price",
                                config={
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                                },
                            )
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="graph",
                                config={
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                                },
                            ),
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dt.DataTable(
                                id="info",
                                style_table={"height": "auto"},
                                style_cell={
                                    "white_space": "normal",
                                    "height": "auto",
                                    "backgroundColor": colors["background"],
                                    "color": "white",
                                    "font_size": "16px",
                                },
                                style_data={"border": "#4d4d4d"},
                                style_header={
                                    "backgroundColor": colors["background"],
                                    "fontWeight": "bold",
                                    "border": "#4d4d4d",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "center"}
                                    for c in ["attribute", "value"]
                                ],
                            ),
                            width={"size": 6, "offset": 3},
                        )
                    ]
                ),
            ]
        ),
        dcc.Store(id='xaxis-range', data={'start': None, 'end': None}),  # Store the x-axis range
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds 
            n_intervals=0
        )
    ],
)
# Initialize a global variable to store historical data
global_df = pd.DataFrame()

# model_helper = ModelHelper(model_type='LSTM')
# model_helper = ModelHelper(model_type='RNN')
model_helper = ModelHelper(model_type='XGBoost')

@app.callback(
    [Output("graph", "figure"), Output("live_price", "figure"), Output('xaxis-range', 'data')],
    [Input("submit-button-state", "n_clicks"), Input('interval-component', 'n_intervals')],
    [State("crypto_pair", "value"), State("chart", "value"), State('xaxis-range', 'data')]
)
def graph_generator(n_clicks, n_intervals, pair, chart_name, xaxis_range):
    global global_df
    
    # Fetch new data
    if n_intervals > 0:
        global_df = DataUtils.update_df(pair, TimeFrame.day, global_df)
    else:
        global_df = DataUtils.init_df(pair, TimeFrame.day)
    
    df = global_df.copy()
    print("\n....> Data: ")
    print(df)
    
    if df.empty:
        return {}, {}, xaxis_range
    
    # Sorting the data
    df.sort_index(inplace=True)

    # Generate predictions
    predictions, future_dates, future_predictions = model_helper.process_and_predict(df, num_days= 30)
    
    # Add actual and predicted prices to the graph
    data = []
    if chart_name == "Line":
        data.append(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close', line=dict(color='blue')))
    elif chart_name == "Candlestick":
        data.append(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
    elif chart_name == "Line and Candlestick":
        data.append(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
        data.append(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close', line=dict(color='blue')))

    # Add current predicted prices
    data.append(go.Scatter(x=df.index, y=predictions, mode='lines', name='Predicted', line=dict(color='green')))
    
    # Add future predicted prices
    data.append(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Prediction', line=dict(color='red')))

    fig = {
        'data': data,
        'layout': go.Layout(
            title=chart_name,
            height=1000,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
            xaxis=dict(
                range=[df.index.min(), future_dates.max()],
                title='Time',
                tickformat='%Y-%m-%d %H:%M',
                rangeslider=dict(visible=False),
            ),
            yaxis={'title': 'Price'},
        )
    }
    
    # Live price figure
    live_price_fig = {
        "data": [
            go.Indicator(
                mode="number+delta",
                value=float(df['close'].iloc[-1]),
                number={
                    "prefix": "$",
                    "valueformat": ".2f"  
                },
                delta={
                    "position": "bottom",
                    "reference": float(df['close'].iloc[-2]),
                },
            )
        ],
        "layout": go.Layout(
            title={"text": f"Live {pair} Price", "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            font=dict(color=colors['text']),
            paper_bgcolor=colors['background'],
            height=250,
        ),
    }

    # Update xaxis range data
    xaxis_range = {'start': df.index.min(), 'end': future_dates.max()}
    
    return fig, live_price_fig, xaxis_range

if __name__ == "__main__":
    app.run_server(debug=True)
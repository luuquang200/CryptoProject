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
                            width={"size": 3, "offset": 1},
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
                                value="Candlestick",
                                style={"color": "#000000"},
                            ),
                            width={"size": 2},
                        ),
                        dbc.Col(  # Model type
                            dcc.Dropdown(
                                id="model_type",
                                options=[
                                    {"label": "LSTM", "value": "LSTM"},
                                    {"label": "RNN", "value": "RNN"},
                                    {"label": "XGBoost", "value": "XGBoost"},
                                ],
                                value="XGBoost",
                                style={"color": "#000000"},
                            ),
                            width={"size": 2},
                        ),
                        dbc.Col(  # Display days
                            dcc.Dropdown(
                                id="display_days",
                                options=[
                                    {"label": "1 month", "value": 30},
                                    {"label": "3 months", "value": 90},
                                    {"label": "6 months", "value": 180},
                                    {"label": "100 days", "value": 100},
                                    {"label": "All", "value": "All"},
                                ],
                                value=100,
                                style={"color": "#000000"},
                            ),
                            width={"size": 2},
                        ),
                        dbc.Col(  # Button
                            dbc.Button(
                                "Plot",
                                id="submit-button-state",
                                className="mr-1",
                                n_clicks=0,
                            ),
                            width={"size": 1},
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
        dcc.Store(id='selected-pair', data='BTCUSDT'),
        dcc.Interval(
            id='interval-component',
            interval=3*1000,  # in milliseconds 
            n_intervals=0
        )
    ],
)


# Initialize a global variable to store historical data
global_df = pd.DataFrame()

LSTM_model = ModelHelper(model_type='LSTM')
RNN_model = ModelHelper(model_type='RNN')
XGBoost_model = ModelHelper(model_type='XGBoost')

def switch_model(model_type):
    if model_type == 'LSTM':
        return LSTM_model
    elif model_type == 'RNN':
        return RNN_model
    elif model_type == 'XGBoost':
        return XGBoost_model
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")


@app.callback(
    [Output("graph", "figure"), Output("live_price", "figure"), Output('xaxis-range', 'data'), Output('selected-pair', 'data')],
    [Input("submit-button-state", "n_clicks"), Input('interval-component', 'n_intervals')],
    [State("crypto_pair", "value"), State("chart", "value"), State("model_type", "value"), State("display_days", "value"), State('xaxis-range', 'data'), State('selected-pair', 'data')]
)
def graph_generator(n_clicks, n_intervals, pair, chart_name, model_type, display_days, xaxis_range, selected_pair):
    global global_df

    # Reset global_df if the selected pair has changed
    if pair != selected_pair:
        global_df = pd.DataFrame()

    # Update model_helper based on selected model type
    model_helper = switch_model(model_type)

    # Fetch new data
    if global_df.empty:
        print("Initializing data...")
        global_df = DataUtils.init_df(pair, TimeFrame.day)
    else:
        if n_intervals > 0:
            print("Updating data...")
            global_df = DataUtils.update_df(pair, TimeFrame.day, global_df)

    df = global_df.copy()
    print("\n....> Data: ")
    print(df)

    if df.empty:
        return {}, {}, xaxis_range, pair

    # Sorting the data
    df.sort_index(inplace=True)

    # Limit the data based on the selected display_days
    if display_days != "All":
        df_display = df[-int(display_days):]
    else:
        df_display = df

    # Generate predictions
    predictions, future_dates, future_predictions = model_helper.process_and_predict(df, num_days=30)

    # Add actual and predicted prices to the graph
    data = []
    if chart_name == "Line":
        data.append(go.Scatter(x=df_display.index, y=df_display['close'], mode='lines', name='Close', line=dict(color='blue')))
    elif chart_name == "Candlestick":
        data.append(go.Candlestick(x=df_display.index, open=df_display['open'], high=df_display['high'], low=df_display['low'], close=df_display['close'], name='Candlestick'))
    elif chart_name == "Line and Candlestick":
        data.append(go.Candlestick(x=df_display.index, open=df_display['open'], high=df_display['high'], low=df_display['low'], close=df_display['close'], name='Candlestick'))
        data.append(go.Scatter(x=df_display.index, y=df_display['close'], mode='lines', name='Close', line=dict(color='blue')))

    # Add current predicted prices
    if display_days != "All":
        data.append(go.Scatter(x=df_display.index, y=predictions[-int(display_days):], mode='lines', name='Predicted', line=dict(color='purple')))
    else:
        data.append(go.Scatter(x=df_display.index, y=predictions, mode='lines', name='Predicted', line=dict(color='purple')))

    # Add future predicted prices
    data.append(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Prediction', line=dict(color='yellow')))

    fig = {
        'data': data,
        'layout': go.Layout(
            title=chart_name,
            height=1000,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
            xaxis=dict(
                range=[df_display.index.min(), future_dates.max()],
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
    xaxis_range = {'start': df_display.index.min(), 'end': future_dates.max()}

    return fig, live_price_fig, xaxis_range, pair

if __name__ == "__main__":
    app.run_server(debug=True)

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
from BB import BollingerBands
from CNN import TradingSignalCNN
from MACD import MovingAverageConvergenceDivergence
from ROC import RateOfChange
from RS import ResistanceSupport
from RSI import RelativeStrengthIndex
from data_utils import DataUtils, TimeFrame
from model_helper import ModelHelper
import os
from MA import MovingAverages

# Load environment variables từ file .env
load_dotenv()

# Define style color
colors = {"background": "#191d21", "text": "#ffffff"}

# List of crypto pairs to display
crypto_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]

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
                                            "margin-top": "50px",
                                            "margin-bottom": "50px",
                                        },
                                    )
                                ]
                            )
                        )
                    ]
                )
            ]
        ),
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
                            width={"size": 2, "offset": 1},
                        ),
                        dbc.Col(  # Graph type
                            dcc.Dropdown(
                                id="chart",
                                options=[
                                    {"label": "line", "value": "Line"},
                                    {"label": "candlestick", "value": "Candlestick"},
                                    {"label": "line and candlestick", "value": "Line and Candlestick"},
                                ],
                                value="Candlestick",
                                style={"color": "#000000"},
                                placeholder="Select chart type",
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
                                    {"label": "Transformer", "value": "Transformer"},
                                ],
                                value="LSTM",
                                style={"color": "#000000"},
                                placeholder="Select model type",
                            ),
                            
                            width={"size": 2},
                        ),
                         dbc.Col(  # Technical indicator
                            dcc.Dropdown(
                                id="technical_indicator",
                                options=[
                                    {"label": "Moving average", "value": "MA"},
                                    {"label": "Bollinger Bands", "value": "BB"},
                                    {"label": "Relative Strength Index", "value": "RSI"},
                                    {"label": "Rate of Change", "value": "ROC"},
                                    {"label": "Resistance and Support", "value": "RS"},
                                    {"label": "Moving Average Convergence Divergence", "value": "MACD"},
                                    {"label": "CNN", "value": "CNN"},
                                ],
                                value="None",
                                placeholder="Select technical indicator",
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
                                placeholder="Select display days",
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
                ),
                dbc.Row(
                    [
                        dbc.Col(  # MA types
                            dcc.Dropdown(
                                id='ma-type-dropdown',
                                options=[{'label': ma, 'value': ma} for ma in ['SMA', 'EMA', 'WMA', 'VWMA']],
                                value='SMA',
                                style={"width": "100px", "margin": "16px auto", "display": "none"},
                                clearable=False
                            ),
                            width={"size": 2, "offset": 1},
                        ),
                        dbc.Col(  # Short-term MA period
                            dcc.Dropdown(
                                id='period1-dropdown',
                                options=[{'label': str(i), 'value': i} for i in [15, 20, 30]],
                                value=20,
                                style={"width": "100px", "margin": "16px auto", "display": "none"},
                                clearable=False
                            ),
                            width={"size": 2},
                            
                        ),
                        dbc.Col(  # Medium-term MA period
                            dcc.Dropdown(
                                id='period2-dropdown',
                                options=[{'label': str(i), 'value': i} for i in [50, 80, 100]],
                                value=50,
                                style={"width": "100px", "margin": "16px auto", "display": "none"},
                                clearable=False
                            ),
                            width={"size": 2},
                        ),
                         dbc.Col(  # Long-term MA period
                            dcc.Dropdown(
                                id='period3-dropdown',
                                options=[{'label': str(i), 'value': i} for i in [120, 150, 200]],
                                value=200,
                                style={"width": "100px", "margin": "16px auto", "display": "none"},
                                clearable=False
                            ),
                            width={"size": 2},
                        ),
                    ],
                    justify="center",
                    align="center",
                ),
                
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
Transformer_model = ModelHelper(model_type='Transformer')
CNN_model = TradingSignalCNN()


def switch_model(model_type):
    if model_type == 'LSTM':
        return LSTM_model
    elif model_type == 'RNN':
        return RNN_model
    elif model_type == 'XGBoost':
        return XGBoost_model
    elif model_type == 'Transformer':
        return Transformer_model
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

@app.callback(
    [
        Output("graph", "figure"), 
        Output("live_price", "figure"), 
        Output('xaxis-range', 'data'), Output('selected-pair', 'data')
    ],
    [
        Input("submit-button-state", "n_clicks"), 
        Input('interval-component', 'n_intervals')
    ],
    [
        State("crypto_pair", "value"), 
        State("chart", "value"), 
        State("model_type", "value"), 
        State("display_days", "value"), 
        State('xaxis-range', 'data'), 
        State('selected-pair', 'data'), 
        State('technical_indicator', 'value'), 
        State('ma-type-dropdown', 'value'), 
        State('period1-dropdown', 'value'), 
        State('period2-dropdown', 'value'), 
        State('period3-dropdown', 'value')
    ]
)
def graph_generator(n_clicks, n_intervals, pair, chart_name, model_type, display_days, xaxis_range, selected_pair, technical_indicator, ma_type, period1, period2, period3):
    global global_df

    # Reset global_df if the selected pair has changed
    if pair != selected_pair:
        global_df = pd.DataFrame()

    # Update model_helper based on selected model type
    model_helper = switch_model(model_type)

    global_df = load_and_update_data(pair, model_type)
    df = global_df.copy()

    if df.empty:
        return {}, {}, xaxis_range, pair

    df.sort_index(inplace=True)

    if display_days != "All":
        df_display = df[-int(display_days):]
    else:
        df_display = df

    predictions, future_dates, future_predictions = process_and_predict(model_helper, df, pair, num_days=30)

    # Initialize the figure
    fig = go.Figure()

    # Add actual and predicted prices to the graph
    if chart_name == "Line":
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['close'], mode='lines', name='Close', line=dict(color='white')))
        # add mode markers+text to show the value of the last close price
        close_value = float(df_display['close'].iloc[-1])  # Convert the string to float before formatting
        fig.add_trace(go.Scatter(x=[df_display.index[-1]], y=[close_value], mode='markers+text', name='Close Value',
                                 text=[f"${close_value:.2f}"], textposition="top right", marker=dict(color='white')))
    elif chart_name == "Candlestick":
        fig.add_trace(go.Candlestick(x=df_display.index, open=df_display['open'], high=df_display['high'], low=df_display['low'], close=df_display['close'], name='Candlestick'))
        # add mode markers+text to show the value of the last close price
        close_value = float(df_display['close'].iloc[-1])  # Convert the string to float before formatting
        fig.add_trace(go.Scatter(x=[df_display.index[-1]], y=[close_value], mode='markers+text', name='Close Value',
                                 text=[f"${close_value:.2f}"], textposition="top right", marker=dict(color='white')))
    elif chart_name == "Line and Candlestick":
        fig.add_trace(go.Candlestick(x=df_display.index, open=df_display['open'], high=df_display['high'], low=df_display['low'], close=df_display['close'], name='Candlestick'))
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['close'], mode='lines', name='Close', line=dict(color='white')))

    # Add current predicted prices
    if display_days != "All":
        fig.add_trace(go.Scatter(x=df_display.index, y=predictions[-int(display_days):], mode='lines', name='Predicted', line=dict(color='purple')))
    else:
        fig.add_trace(go.Scatter(x=df_display.index, y=predictions, mode='lines', name='Predicted', line=dict(color='purple')))
    # add mode markers to show the value of the last predicted price
    fig.add_trace(go.Scatter(x=[df_display.index[-1]], y=[predictions[-1]], mode='markers', name='Predicted Value', marker=dict(color='purple')))

    # Add future predicted prices
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Prediction', line=dict(color='yellow')))
    # add mode markers+text to show the value of the first future prediction
    fig.add_trace(go.Scatter(x=[future_dates[0]], y=[future_predictions[0]], mode='markers+text', name='Future Prediction Value',
                             text=[f"${future_predictions[0]:.2f}"], textposition="top right", marker=dict(color='yellow')))

    fig.update_layout(
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
            showgrid=False,
        ),
        yaxis=dict(
            title='Price',
            showgrid=False  # Remove the grid from the y-axis
        ),
    )

    # Live price figure
    live_price_fig = go.Figure(
        data=[
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
    )
    live_price_fig.update_layout(
        title={"text": f"Live {pair} Price", "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        font=dict(color=colors['text']),
        paper_bgcolor=colors['background'],
        height=250,
    )

    if technical_indicator == 'MA':
        MovingAverages.add_trading_signals(df_display, MA_type=ma_type, period1=period1, period2=period2, period3=period3)
        fig = MovingAverages.add_trace_to_plot(fig, df_display, period1=period1, period2=period2, period3=period3)
    elif technical_indicator == 'RSI':
        # Adding a secondary y-axis for RSI
        fig.update_layout(
            yaxis2=dict(
                title="RSI",
                overlaying='y',
                side='right',
                range=[0, 100],  # RSI values range between 0 and 100
                showgrid=False
            )
        )
        df_display = RelativeStrengthIndex.add_RSI_signals(df_display, period=14)
        fig = RelativeStrengthIndex.add_RSI_trace(fig, df_display, period=14)
        fig = RelativeStrengthIndex.add_RSI_signal_trace(fig, df_display, period=14)
    elif technical_indicator == 'BB':
        df_display = BollingerBands.add_BB_signals(df_display, period=20)
        fig = BollingerBands.add_BB_trace(fig, df_display, period=20)
        fig = BollingerBands.add_BB_signal_trace(fig, df_display)
    elif technical_indicator == 'ROC':
        # Adding a secondary y-axis for ROC
        fig.update_layout(
            yaxis2=dict(
                title="ROC",
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
        df_display = RateOfChange.add_ROC_signals(df_display, period=14)
        fig = RateOfChange.add_ROC_trace(fig, df_display, period=14)
        fig = RateOfChange.add_ROC_signal_trace(fig, df_display)
    elif technical_indicator == 'RS':
        df_display = ResistanceSupport.calculate_resistance_support(df_display, period=20)
        fig = ResistanceSupport.add_resistance_support_trace(fig, df_display, period=20)
    elif technical_indicator == 'MACD':
        # Adding a secondary y-axis for MACD
        fig.update_layout(
            yaxis2=dict(
                title="MACD",
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
        # Assuming df_display is your DataFrame with the stock data
        df_display = MovingAverageConvergenceDivergence.calculate_macd(df_display, short_period=12, long_period=26, signal_period=9, column='close')
        # Add MACD traces to the figure
        fig = MovingAverageConvergenceDivergence.add_macd_trace(fig, df_display)
    elif technical_indicator == 'CNN':
        df_display = CNN_model.generate_signals(df_display)
        fig = CNN_model.add_signal_trace(fig, df_display)



    # Update xaxis range data
    xaxis_range = {'start': df_display.index.min(), 'end': future_dates.max()}

    return fig, live_price_fig, xaxis_range, pair


def load_data_from_csv(pair, model_type):
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    filename = os.path.join(cache_dir, f"{pair}-{model_type}.csv")
    if os.path.isfile(filename):
        df = pd.read_csv(filename, parse_dates=['timestamp'], index_col='timestamp')
        return df
    else:
        return pd.DataFrame()

def save_data_to_csv(df, pair, model_type):
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    filename = os.path.join(cache_dir, f"{pair}-{model_type}.csv")
    df.to_csv(filename)


def load_and_update_data(pair, model_type):
    df = load_data_from_csv(pair, model_type)
    if df.empty:
        print("Initializing data...")
        df = DataUtils.init_df(pair, TimeFrame.day)
    else:
        print("Updating data...")
        df = DataUtils.update_df(pair, TimeFrame.day, df)
    save_data_to_csv(df, pair, model_type)  
    return df


def process_and_predict(model_helper, df, pair, num_days=30):
    if df.empty:
        print("DataFrame is empty. Initializing with default data.")
        df = DataUtils.init_df(pair, TimeFrame.day)
    
    if 'predictions' not in df.columns:
        print("Calculating predictions...")
        if model_helper.model_type == 'XGBoost':
            predictions, future_dates, future_predictions = model_helper.generate_predictions(df, model_helper.model)
        else:
            predictions, future_dates, future_predictions = model_helper.process_and_predict(df, num_days=num_days)
        df['predictions'] = predictions
        save_data_to_csv(df, pair, model_helper.model_type)
    else:
        print("Loading predictions from CSV...")
        predictions = df['predictions'].values
        if model_helper.model_type == 'XGBoost':
            predictions, future_dates, future_predictions = model_helper.generate_predictions(df, model_helper.model)
        else:
            last_days_for_future = model_helper.preprocess_data(df)[1]
            future_dates, future_predictions = model_helper.predict_future_prices(last_days_for_future, num_days=num_days)
    return predictions, future_dates, future_predictions

if __name__ == "__main__":
    app.run_server(debug=True)

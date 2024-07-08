import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table as dt
import plotly.graph_objs as go
from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from data_utils import DataUtils, TimeFrame
import plotly.graph_objs as go

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
            interval=3*1000,  # in milliseconds 
            n_intervals=0
        )
    ],
)

# Initialize a global variable to store historical data
global_df = pd.DataFrame()

@app.callback(
    [Output("graph", "figure"), Output("live_price", "figure"), Output('xaxis-range', 'data')],
    [Input("submit-button-state", "n_clicks"), Input('interval-component', 'n_intervals')],
    [State("crypto_pair", "value"), State("chart", "value"), State('xaxis-range', 'data')]
)
def graph_generator(n_clicks, n_intervals, pair, chart_name, xaxis_range):
    global global_df
    
    # Fetch new data every 10 seconds
    if n_intervals > 0:
        global_df = DataUtils.update_df(pair, TimeFrame.day, global_df)
    else:
        global_df = DataUtils.init_df(pair, TimeFrame.day)
    
    df = global_df.copy()
    
    # Kiểm tra số lượng dòng
    print(f"---> Number of rows: {len(df)}")
    print(df.tail())
    if df.empty:
        print("Dataframe is empty!")
        return None
    
    # Sorting the data
    df.sort_index(inplace=True)

    # Đảm bảo có ít nhất 60 ngày dữ liệu
    if len(df) < 60:
        print("Not enough data for prediction!")
        return None
    
    # Load mô hình LSTM
    model_lstm = load_model('LSTM.keras', compile=False)
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    print(f"Size of scaled_data: {scaled_data.shape}")

    # Bắt đầu với 60 ngày cuối cùng của dữ liệu để dự đoán
    last_60_days = scaled_data[-60:].reshape(1, 60, 1)

    # Dự đoán giá cho đến tháng 12 năm 2025
    future_dates = pd.date_range(start=df.index.max() + timedelta(days=1), end='2024-7-31', freq='B')
    future_predictions = []

    for date in future_dates:
        predicted_price = model_lstm.predict(last_60_days)
        future_predictions.append(predicted_price[0, 0])
        print(f"Predicted price for {date}: {predicted_price[0, 0]}")
        # Thêm giá dự đoán vào dữ liệu đầu vào và duy trì hình dạng
        predicted_price_reshaped = predicted_price.reshape(1, 1, 1)
        last_60_days = np.append(last_60_days[:, 1:, :], predicted_price_reshaped, axis=1)

    # Inverse transform the predicted prices
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Thêm đường dự đoán vào biểu đồ
    data = []

    # Selecting graph type
    if chart_name == "Line":
        data.append(go.Scatter(x=df.index, y=df['close'], mode='lines', name='close', line=dict(color='blue')))
    elif chart_name == "Candlestick":
        data.append(go.Candlestick(x=df.index,
                                   open=df['open'],
                                   high=df['high'],
                                   low=df['low'],
                                   close=df['close'],
                                   name='candlestick'))
    elif chart_name == "Line and Candlestick":
        data.append(go.Candlestick(x=df.index,
                                   open=df['open'],
                                   high=df['high'],
                                   low=df['low'],
                                   close=df['close'],
                                   name='candlestick'))
        data.append(go.Scatter(x=df.index, y=df['close'], mode='lines', name='close', line=dict(color='blue')))

    # Simple Moving Average
    if chart_name == "SMA":
        close_ma_10 = df.close.rolling(10).mean()
        close_ma_15 = df.close.rolling(15).mean()
        close_ma_30 = df.close.rolling(30).mean()
        close_ma_100 = df.close.rolling(100).mean()
        data.extend([
            go.Scatter(
                x=list(close_ma_10.index), y=list(close_ma_10), name="10 Days"
            ),
            go.Scatter(
                x=list(close_ma_15.index), y=list(close_ma_15), name="15 Days"
            ),
            go.Scatter(
                x=list(close_ma_30.index), y=list(close_ma_30), name="30 Days"
            ),
            go.Scatter(
                x=list(close_ma_100.index), y=list(close_ma_100), name="100 Days"
            ),
        ])

    # Add the prediction line to the graph
    data.append(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='LSTM Prediction', line=dict(color='red')))

    fig = {
        'data': data,
        'layout': go.Layout(
            title= chart_name,
            height=1000,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
            xaxis=dict(
                range=[df.index.min(), future_dates.max()],
                title='Time',
                tickformat='%Y-%m-%d %H:%M',
                rangeslider=dict( visible=False ),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]),
                    bgcolor=colors["background"],
                    activecolor="blue",
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ),
            yaxis=dict(
                title='Price (USDT)'
            )
        )
    }

    # Fetching live price
    price = DataUtils.get_current_price(pair)
    prev_close = df['close'].iloc[-1]

    live_price = go.Figure(
        data=[
            go.Indicator(
                domain={"x": [0, 1], "y": [0, 1]},
                value=float(price),
                mode="number+delta",
                title={"text": "Price"},
                delta={"reference": float(prev_close)},
            )
        ],
        layout={
            "height": 300,
            "showlegend": True,
            "plot_bgcolor": colors["background"],
            "paper_bgcolor": colors["background"],
            "font": {"color": colors["text"]},
        },
    )

    # Store the current x-axis range
    xaxis_range = {'start': df.index.min(), 'end': df.index.max()}

    return fig, live_price, xaxis_range

if __name__ == "__main__":
    app.run_server(debug=True)

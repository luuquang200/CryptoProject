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
import pytz
from dotenv import load_dotenv
from utils import DataUtils, TimeFrame
# Load environment variables from .env file
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
            interval=2*1000,  # in milliseconds 
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
        global_df = DataUtils.init_df(pair, TimeFrame.minute)
    else:
        global_df = DataUtils.update_df(pair, TimeFrame.minute, global_df)

    df = global_df.copy()
    
    # print number of rows
    print (f"Number of rows: {len(df)}")
    print (df.tail())
    if df.empty:
        print("Dataframe is empty!")
    
    # Sorting the data
    df.sort_index(inplace=True)
    
    if xaxis_range['start'] is None or xaxis_range['end'] is None:
        x_range = [min(df.index), max(df.index)]
    else:
        x_range = [xaxis_range['start'], xaxis_range['end']]

    # Selecting graph type
    if chart_name == "Line":
        fig = {
            'data': [go.Scatter(x=df.index, y=df['close'], mode='lines', name='close')],
            'layout': go.Layout(
                title= chart_name,
                height=1000,
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']},
                xaxis=dict(
                    range=x_range,
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

    elif chart_name == "Candlestick":

        fig = {
            'data': [go.Candlestick(x=df.index,
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='candlestick')],
            'layout': go.Layout( 
                title= chart_name,
                height=1000,
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']},
                xaxis=dict(
                    range=x_range,
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
       

    # Simple Moving Average
    if chart_name == "SMA":
        close_ma_10 = df.close.rolling(10).mean()
        close_ma_15 = df.close.rolling(15).mean()
        close_ma_30 = df.close.rolling(30).mean()
        close_ma_100 = df.close.rolling(100).mean()
        fig = go.Figure(
            data=[
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
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
            },
        )


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

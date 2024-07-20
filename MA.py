import numpy as np
import pandas as pd
import plotly.graph_objs as go

class MovingAverages:
    @staticmethod
    def SMA(df, period=50, column='close'):
        return df[column].rolling(window=period).mean()

    @staticmethod
    def EMA(df, period=50, column='close'):
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def WMA(df, period=50, column='close'):
        weights = np.arange(1, period + 1)
        return df[column].rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    @staticmethod
    def VWMA(df, period=50, column='close'):
        volume = df['Vol.'] if 'Vol.' in df.columns else pd.Series(np.ones(len(df)), index=df.index)
        return (df[column] * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

    @staticmethod
    def MA(df, period=30, column='close', ma_type="SMA"):
        if ma_type == "SMA":
            return MovingAverages.SMA(df, period, column)
        elif ma_type == "EMA":
            return MovingAverages.EMA(df, period, column)
        elif ma_type == "WMA":
            return MovingAverages.WMA(df, period, column)
        elif ma_type == "VWMA":
            return MovingAverages.VWMA(df, period, column)
        else:
            raise ValueError("Invalid ma_type. Use 'SMA', 'EMA', 'WMA', or 'VWMA'.")
        
    @staticmethod
    def add_trading_signals(df, MA_type="SMA", period1=20, period2=50 ,period3=200):
        df['line1'] = MovingAverages.MA(df, period=period1, column='close', ma_type=MA_type)
        df['line2'] = MovingAverages.MA(df, period=period2, column='close', ma_type=MA_type)
        df['line3'] = MovingAverages.MA(df, period=period3, column='close', ma_type=MA_type)

        # Condition 1
        df['Signal'] = np.where(df["line1"] > df["line2"], 1, 0)
        df['Position'] = df['Signal'].diff()

        df['Buy'] = np.where(df['Position'] == 1, df['close'], np.nan)
        df['Sell'] = np.where(df['Position'] == -1, df['close'], np.nan)

        # Condition 2
        df['Golden_Signal'] = np.where(df["line2"] > df["line3"], 1, 0)
        df['Golden_Position'] = df['Golden_Signal'].diff()

        df['Golden_Buy'] = np.where(df['Golden_Position'] == 1, df['close'], np.nan)
        df['Death_Sell'] = np.where(df['Golden_Position'] == -1, df['close'], np.nan)
        return df
    
    @staticmethod
    def add_trace_to_plot(fig, df, period1=20, period2=50, period3=200):
        # Short-term MA
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['line1'],
                                mode='lines',
                                name=f'Short-term MA {period1}',
                                line=dict(color='royalblue')))

        # Medium-term MA
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['line2'],
                                mode='lines',
                                name=f'Medium-term MA {period2}',
                                line=dict(color='darkorange')))

        # Long-term MA
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['line3'],
                                mode='lines',
                                name=f'Long-term MA {period3}',
                                line=dict(color='seagreen')))

        # Buy Signal
        fig.add_trace(go.Scatter(x=df.index[df['Position'] == 1],
                                y=df['close'][df['Position'] == 1],
                                mode='markers',
                                marker=dict(symbol='triangle-up', color='green', size=32, opacity=0.8, line=dict(width=2, color='darkgreen')),
                                name='Buy Signal'))

        # Sell Signal
        fig.add_trace(go.Scatter(x=df.index[df['Position'] == -1],
                                y=df['close'][df['Position'] == -1],
                                mode='markers',
                                marker=dict(symbol='triangle-down', color='red', size=32, opacity=0.8, line=dict(width=2, color='darkred')),
                                name='Sell Signal'))

        # Golden Buy Signal
        fig.add_trace(go.Scatter(x=df.index[df['Golden_Position'] == 1],
                                y=df['close'][df['Golden_Position'] == 1],
                                mode='markers',
                                marker=dict(symbol='triangle-up', color='gold', size=32, opacity=0.8, line=dict(width=2, color='darkgoldenrod')),
                                name='Golden Buy Signal',
                                visible='legendonly'))

        # Death Sell Signal
        fig.add_trace(go.Scatter(x=df.index[df['Golden_Position'] == -1],
                                y=df['close'][df['Golden_Position'] == -1],
                                mode='markers',
                                marker=dict(symbol='triangle-down', color='maroon', size=32, opacity=0.8, line=dict(width=2, color='darkred')),
                                name='Death Sell Signal',
                                visible='legendonly'))
        return fig

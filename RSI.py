import numpy as np
import pandas as pd
import plotly.graph_objs as go

class RelativeStrengthIndex:
    @staticmethod
    def calculate_RSI(df, period=14, column='close'):
        # Ensure the 'close' column is numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # Calculate the differences
        delta = df[column].diff()

        # Calculate gain and loss
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero
        loss = loss.replace(0, np.nan)

        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))

        # Replace NaN with 0 for initial values
        RSI = RSI.fillna(0)

        df[f'RSI_{period}'] = RSI
        return df

    @staticmethod
    def add_RSI_trace(fig, df, period=14):
        # Add RSI line
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df[f'RSI_{period}'],
                                 mode='lines',
                                 name=f'RSI {period}',
                                 line=dict(color='blue'),
                                 yaxis='y2'))

        # Add overbought/oversold lines
        fig.add_trace(go.Scatter(x=[df.index.min(), df.index.max()],
                                 y=[70, 70],
                                 mode='lines',
                                 name='Overbought',
                                 line=dict(color='red', dash='dash'),
                                 yaxis='y2'))

        fig.add_trace(go.Scatter(x=[df.index.min(), df.index.max()],
                                 y=[30, 30],
                                 mode='lines',
                                 name='Oversold',
                                 line=dict(color='green', dash='dash'),
                                 yaxis='y2'))
        
        return fig

    @staticmethod
    def add_RSI_signals(df, period=14):
        df = RelativeStrengthIndex.calculate_RSI(df, period)
        df['RSI_Buy'] = np.where(df[f'RSI_{period}'] < 30, df['close'], np.nan)
        df['RSI_Sell'] = np.where(df[f'RSI_{period}'] > 70, df['close'], np.nan)
        return df
    
    @staticmethod
    def add_RSI_signal_trace(fig, df, period=14):
        # Add RSI Buy Signal
        fig.add_trace(go.Scatter(x=df.index[df[f'RSI_Buy'].notnull()],
                                 y=df[f'RSI_Buy'][df[f'RSI_Buy'].notnull()],
                                 mode='markers',
                                 marker=dict(symbol='triangle-up', color='green', size=32, opacity=0.8, line=dict(width=2, color='darkgreen')),
                                 name='RSI Buy Signal'))

        # Add RSI Sell Signal
        fig.add_trace(go.Scatter(x=df.index[df[f'RSI_Sell'].notnull()],
                                 y=df[f'RSI_Sell'][df[f'RSI_Sell'].notnull()],
                                 mode='markers',
                                 marker=dict(symbol='triangle-down', color='red', size=32, opacity=0.8, line=dict(width=2, color='darkred')),
                                 name='RSI Sell Signal'))
        
        return fig

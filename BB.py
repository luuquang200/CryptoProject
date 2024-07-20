import numpy as np
import pandas as pd
import plotly.graph_objs as go

class BollingerBands:
    @staticmethod
    def calculate_BB(df, period=20, column='close', num_std=2):
        # Ensure the 'close' column is numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')

        df['SMA'] = df[column].rolling(window=period).mean()
        df['STD'] = df[column].rolling(window=period).std()
        
        df['Upper_Band'] = df['SMA'] + (df['STD'] * num_std)
        df['Lower_Band'] = df['SMA'] - (df['STD'] * num_std)
        
        return df

    @staticmethod
    def add_BB_signals(df, period=20, column='close', num_std=2):
        df = BollingerBands.calculate_BB(df, period, column, num_std)
        
        df['Buy'] = np.where(df[column] < df['Lower_Band'], df[column], np.nan)
        df['Sell'] = np.where(df[column] > df['Upper_Band'], df[column], np.nan)
        
        return df
    
    @staticmethod
    def add_BB_trace(fig, df, period=20):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Upper_Band'],
            mode='lines',
            name=f'Upper Band {period}',
            line=dict(color='orange'),
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Lower_Band'],
            mode='lines',
            name=f'Lower Band {period}',
            line=dict(color='orange'),
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA'],
            mode='lines',
            name=f'SMA {period}',
            line=dict(color='blue')
        ))
        
        return fig

    @staticmethod
    def add_BB_signal_trace(fig, df):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Buy'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=32, opacity=0.8),
            name='Buy Signal'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Sell'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=32, opacity=0.8),
            name='Sell Signal'
        ))

        return fig

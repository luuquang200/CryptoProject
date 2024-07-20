import pandas as pd
import numpy as np
import plotly.graph_objs as go

class RateOfChange:
    @staticmethod
    def calculate_ROC(df, period=14, column='close'):
        # Ensure the 'close' column is numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')

        df[f'ROC_{period}'] = df[column].pct_change(periods=period) * 100
        return df

    @staticmethod
    def add_ROC_signals(df, period=14, column='close'):
        df = RateOfChange.calculate_ROC(df, period, column)
        
        df['Buy'] = np.where(df[f'ROC_{period}'] > 0, df[column], np.nan)
        df['Sell'] = np.where(df[f'ROC_{period}'] < 0, df[column], np.nan)
        
        return df

    @staticmethod
    def add_ROC_trace(fig, df, period=14):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'ROC_{period}'],
            mode='lines',
            name=f'ROC {period}',
            line=dict(color='orange'),
            yaxis='y2'
        ))
        
        return fig

    @staticmethod
    def add_ROC_signal_trace(fig, df):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Buy'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=16, opacity=0.8),
            name='Buy Signal'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Sell'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=16, opacity=0.8),
            name='Sell Signal'
        ))

        return fig

import numpy as np
import pandas as pd
import plotly.graph_objs as go

class ResistanceSupport:
    @staticmethod
    def calculate_resistance_support(df, period=20, column='close'):
        df['Rolling_Max'] = df[column].rolling(window=period).max()
        df['Rolling_Min'] = df[column].rolling(window=period).min()
        return df

    @staticmethod
    def add_resistance_support_trace(fig, df, period=20, yaxis='y'):
        fig.add_trace(go.Scatter(
            x=df.index,  # Use index instead of a specific time column
            y=df['Rolling_Max'],
            mode='lines',
            name=f'Resistance {period}',
            line=dict(color='red', dash='dash'),
            yaxis=yaxis
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,  # Use index instead of a specific time column
            y=df['Rolling_Min'],
            mode='lines',
            name=f'Support {period}',
            line=dict(color='green', dash='dash'),
            yaxis=yaxis
        ))
        
        return fig

import pandas as pd
import plotly.graph_objs as go

class MovingAverageConvergenceDivergence:
    @staticmethod
    def calculate_macd(df, short_period=12, long_period=26, signal_period=9, column='close'):
        # Ensure the 'close' column is numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        df['EMA_short'] = df[column].ewm(span=short_period, adjust=False).mean()
        df['EMA_long'] = df[column].ewm(span=long_period, adjust=False).mean()
        df['MACD'] = df['EMA_short'] - df['EMA_long']
        df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        return df

    @staticmethod
    def add_macd_trace(fig, df, yaxis='y2'):
        fig.add_trace(go.Scatter(
            x=df.index,  # Use index instead of a specific time column
            y=df['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue'),
            yaxis=yaxis
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,  # Use index instead of a specific time column
            y=df['Signal_Line'],
            mode='lines',
            name='Signal Line',
            line=dict(color='orange'),
            yaxis=yaxis
        ))
        
        fig.add_trace(go.Bar(
            x=df.index,  # Use index instead of a specific time column
            y=df['MACD_Histogram'],
            name='MACD Histogram',
            marker=dict(color='green'),
            yaxis=yaxis
        ))
        
        return fig


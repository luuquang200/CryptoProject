import h5py
import numpy as np
import pandas as pd
import plotly.graph_objs as go

class SimpleClassificationNetwork:
    def __init__(self, model_path='simpleclass_train_50.h5'):
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        with h5py.File(self.model_path, 'r') as hf:
            self.predict_val = hf['predict_valid'][:]
            self.ytest = hf['y_test'][:]
        
        # Process the predictions to get binary classification results
        x = np.argmax(self.predict_val, axis=1)
        self.predict_valid = np.zeros(self.predict_val.shape)
        self.predict_valid[x == 0, 0] = 1
        self.predict_valid[x == 1, 1] = 1
        self.predict_valid[x == 2, 2] = 1
    
    def generate_signals(self, df):
        df['Buy'] = np.nan
        df['Sell'] = np.nan
        for i in range(len(self.predict_valid[:len(df)])):
            if self.predict_valid[i][1] == 1:
                df.at[df.index[i], 'Buy'] = df['close'].iloc[i]
            elif self.predict_valid[i][2] == 1:
                df.at[df.index[i], 'Sell'] = df['close'].iloc[i]
        return df
    
    def add_signal_trace(self, fig, df):
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Buy'], 
            mode='markers', 
            name='Buy Signal', 
            marker=dict(color='green', symbol='triangle-up', size=18, opacity=0.8),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Sell'], 
            mode='markers', 
            name='Sell Signal', 
            marker=dict(color='red', symbol='triangle-down', size=18, opacity=0.8),
        ))
        return fig

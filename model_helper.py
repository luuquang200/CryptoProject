
import pandas as pd
import numpy as np
from keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import plotly.graph_objects as go
import pandas as pd
import pandas_ta as ta

class ModelHelper:
    def __init__(self, model_type='LSTM'):
        self.model_type = model_type
        self.model = self.load_model()

    def load_model(self):
        if self.model_type == 'LSTM':
            model = load_model('LSTM.keras', compile=False)
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        elif self.model_type == 'XGBoost':
            model = xgb.Booster()
            model.load_model('XGBoost.json')
            return model
        elif self.model_type == 'RNN':
            model = load_model('RNN.keras', compile=False)
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported.")

    # LSTM and RNN
    def preprocess_data(self, df):
        # Ensure at least 60 days of data
        if len(df) < 60:
            return None, None

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        return scaler, last_60_days

    def predict_future_prices(self, last_60_days, scaler, num_days=30):
        future_dates = pd.date_range(start=datetime.today(), periods=num_days, freq='B')
        future_predictions = []

        for _ in range(num_days):
            predicted_price = self.model.predict(last_60_days)
            future_predictions.append(predicted_price[0, 0])
            predicted_price_reshaped = predicted_price.reshape(1, 1, 1)
            last_60_days = np.append(last_60_days[:, 1:, :], predicted_price_reshaped, axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        return future_dates, future_predictions

    def make_predictions(self, df, scaler):
        # Ensure at least 60 days of data
        if len(df) < 60:
            return df, None

        # Prepare data for prediction
        scaled_data = scaler.transform(df['close'].values.reshape(-1, 1))
        X_test = []
        for i in range(60, len(scaled_data)):
            X_test.append(scaled_data[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict prices
        closing_price = self.model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        # Adjust the length of predictions to match the length of the dataframe
        predictions = np.empty_like(df['close'])
        predictions[:] = np.nan
        predictions[-len(closing_price):] = closing_price.flatten()

        # df['Predictions'] = predictions
        return predictions, X_test[-1:]

    # Main function
    def process_and_predict(self, df, num_days=30):
        print(f"Processing data for {self.model_type} model")
        # Print last 5 rows of the dataframe
        print(df.tail())
        if self.model_type == 'XGBoost':
            return self.generate_predictions(df, self.model)
        else:
            scaler, last_60_days = self.preprocess_data(df)
            if scaler is None or last_60_days is None or last_60_days.size == 0:
                return df, None, None, None
            
            predictions, last_60_days_for_future = self.make_predictions(df, scaler)
            future_dates, future_predictions = self.predict_future_prices(last_60_days_for_future, scaler, num_days=num_days)
            # Print the first 5 predictions
            print(f"First 5 predictions: {future_predictions[:5]}")
            return predictions, future_dates, future_predictions
    
    # Xgboost     
    def add_technical_indicators(self, df):
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['SMA_5'] = ta.sma(df['close'], length=5)
        df['SMA_10'] = ta.sma(df['close'], length=10)
        df['SMA_15'] = ta.sma(df['close'], length=15)
        df['SMA_30'] = ta.sma(df['close'], length=30)
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        return df

    def prepare_data(self, df):
        # Convert to correct data types
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Ensure no NaN values
        df.dropna(inplace=True)
        
        return df

    def calculate_macd(self, data, short_window=12, long_window=26):
        short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=9, adjust=False).mean()
        return macd, signal_line

    def relative_strength_idx(self, df, n=14):
        close = df['close']
        delta = close.diff()
        delta = delta[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def generate_predictions(self, df, model):
        # Ensure close prices are numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        # Prepare the data
        df['EMA_9'] = df['close'].ewm(9).mean()
        df['SMA_5'] = df['close'].rolling(5).mean()
        df['SMA_10'] = df['close'].rolling(10).mean()
        df['SMA_15'] = df['close'].rolling(15).mean()
        df['SMA_30'] = df['close'].rolling(30).mean()
        df['RSI'] = self.relative_strength_idx(df)
        macd, macd_signal = self.calculate_macd(df)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal

        # Select features
        features = ['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal']
        X = df[features]

        dmatrix_test = xgb.DMatrix(X)

        # Predict the closing prices for the test data
        y_pred = model.predict(dmatrix_test)

        # Generate future predictions
        future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
        future_predictions = []

        last_known_data = df.iloc[-1]

        current_date = df.index[-1] + pd.Timedelta(days=1)
        end_date = future_dates[-1]

        while current_date <= end_date:
            # Create a new row with the last known data
            features_values = last_known_data[features].values.reshape(1, -1)
            dmatrix_features = xgb.DMatrix(features_values, feature_names=features)

            # Predict the closing price for the next day
            predicted_close = model.predict(dmatrix_features)[0]
            future_predictions.append(predicted_close)

            # Add the new data point to the dataframe
            new_data_point = {'close': predicted_close}
            new_row = pd.DataFrame([new_data_point], index=[current_date])
            df = pd.concat([df, new_row])

            # Update the technical indicators
            df['EMA_9'] = df['close'].ewm(9).mean()
            df['SMA_5'] = df['close'].rolling(5).mean()
            df['SMA_10'] = df['close'].rolling(10).mean()
            df['SMA_15'] = df['close'].rolling(15).mean()
            df['SMA_30'] = df['close'].rolling(30).mean()
            df['RSI'] = self.relative_strength_idx(df)
            macd, macd_signal = self.calculate_macd(df)
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal

            # Update the last known data
            last_known_data = df.iloc[-1]
            current_date += pd.Timedelta(days=1)
        print(f"First 5 predictions: {future_predictions[:5]}")
        return y_pred, future_dates, future_predictions

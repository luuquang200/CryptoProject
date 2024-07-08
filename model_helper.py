
import pandas as pd
import numpy as np
from keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

class ModelHelper:
    def __init__(self, model_type='LSTM'):
        self.model_type = model_type
        self.model = self.load_model()

    def load_model(self):
        if self.model_type == 'LSTM':
            model = load_model('LSTM.keras', compile=False)
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        # elif self.model_type == 'XGBoost':
        #     pass
        # elif self.model_type == 'RNN':
        #     pass
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported.")

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

        df['Predictions'] = predictions
        return df, X_test[-1:]

    def process_and_predict(self, df, num_days=30):
        scaler, last_60_days = self.preprocess_data(df)
        if scaler is None or last_60_days is None or last_60_days.size == 0:
            return df, None, None, None
        
        df, last_60_days_for_future = self.make_predictions(df, scaler)
        future_dates, future_predictions = self.predict_future_prices(last_60_days_for_future, scaler, num_days=num_days)
        return df, future_dates, future_predictions
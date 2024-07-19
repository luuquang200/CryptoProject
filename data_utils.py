from binance.client import Client
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

class TimeFrame:
    minute = 'minute'
    hour = 'hour'
    day = 'day'

# Binance API credentials from environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize the Binance client
client = Client(API_KEY, API_SECRET)

class DataUtils:
    
    @staticmethod
    def get_historical_klines(symbol, interval, start_str, end_str=None):
        print (f"Fetching data for {symbol} from {start_str} to {end_str}, every {interval}") 

        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df

    @staticmethod
    def get_current_price(symbol):
        price = client.get_symbol_ticker(symbol=symbol)
        return price['price']
    
    @staticmethod
    def init_df(symbol, timeframe):
        if timeframe == TimeFrame.minute:
            return DataUtils.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "2 hours ago UTC")
        elif timeframe == TimeFrame.hour:
            return DataUtils.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 days ago UTC")
        elif timeframe == TimeFrame.day:
            return DataUtils.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "500 days ago UTC")
        else:
            raise ValueError("Invalid timeframe")
        

    @staticmethod
    def update_df(symbol, timeframe, df):
        if timeframe == TimeFrame.minute:
            new_data = DataUtils.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 minute ago UTC")
        elif timeframe == TimeFrame.hour:
            new_data = DataUtils.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 hour ago UTC")
        elif timeframe == TimeFrame.day:
            new_data = DataUtils.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "1 day ago UTC")
        else:
            raise ValueError("Invalid timeframe")
       
        df = pd.concat([df, new_data]).drop_duplicates()
        df = df[~df.index.duplicated(keep='last')]

        return df
    
    @staticmethod
    def get_scaler(pair, df):
        # Extract the 'close' prices
        close_prices = df[['close']].values
        
        # Create and fit the scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(close_prices)
        
        return scaler
    
    

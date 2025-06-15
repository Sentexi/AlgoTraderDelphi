import pandas as pd
import talib

# Function to create lagged features
def create_lagged_features(data, ticker, lag_periods):
    for lag in lag_periods:
        lag_col = f'close_{ticker}_lag{lag}'
        data[lag_col] = data[f'close_{ticker}'].shift(lag)
    return data
        
# Add all technical indicators
def add_technical_indicators(data, ticker):
    close_col = f'close_{ticker}'
    high_col = f'high_{ticker}'
    low_col = f'low_{ticker}'

    data[f'MA50_{ticker}'] = data[close_col].rolling(window=50).mean()
    data[f'EMA20_{ticker}'] = data[close_col].ewm(span=20, adjust=False).mean()
    data[f'RSI_{ticker}'] = talib.RSI(data[close_col], timeperiod=14)

    upper_band, middle_band, lower_band = talib.BBANDS(data[close_col], timeperiod=20)
    data[f'upper_bb_{ticker}'] = upper_band
    data[f'middle_bb_{ticker}'] = middle_band
    data[f'lower_bb_{ticker}'] = lower_band

    data[f'ATR_{ticker}'] = talib.ATR(data[high_col], data[low_col], data[close_col], timeperiod=14)

    macd, macd_signal, _ = talib.MACD(data[close_col], fastperiod=12, slowperiod=26, signalperiod=9)
    data[f'MACD_{ticker}'] = macd
    data[f'MACD_signal_{ticker}'] = macd_signal

    data[f'ADX_{ticker}'] = talib.ADX(data[high_col], data[low_col], data[close_col], timeperiod=14)

    slowk, slowd = talib.STOCH(data[high_col], data[low_col], data[close_col], fastk_period=14, slowk_period=3, slowd_period=3)
    data[f'StochK_{ticker}'] = slowk
    data[f'StochD_{ticker}'] = slowd

    data[f'CCI_{ticker}'] = talib.CCI(data[high_col], data[low_col], data[close_col], timeperiod=20)
    data[f'WilliamsR_{ticker}'] = talib.WILLR(data[high_col], data[low_col], data[close_col], timeperiod=14)

    return data
    
# Load and rename data
def load_and_rename_data(file_path, ticker):
    try:
        df = pd.read_csv(file_path, parse_dates=['datetime'])
        column_mapping = {
            'v': f'volume_{ticker}',
            'vw': f'vw_{ticker}',
            'o': f'open_{ticker}',
            'c': f'close_{ticker}',
            'h': f'high_{ticker}',
            'l': f'low_{ticker}',
            'n': f'trades_{ticker}'
        }
        df.rename(columns=column_mapping, inplace=True)
        selected_columns = ['datetime'] + list(column_mapping.values())
        df = df[selected_columns]
        return df
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None
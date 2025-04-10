import yfinance as yf
import pandas as pd 
import os
import time

import requests
import time
import yfinance as yf

import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def fetch_stock_data(ticker, start_date='2015-01-01', end_date='2024-01-01'):
    """
    Fetch historical stock data for a ticker from Yahoo Finance, along with basic fundamentals.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if stock_data.empty:
        raise ValueError("Downloaded stock data is empty.")
    
    stock = yf.Ticker(ticker)
    # fundamentals = {
    #     "DE Ratio": stock.info.get("debtToEquity"),
    #     "Return on Equity": stock.info.get("returnOnEquity"),
    #     "Price/Book": stock.info.get("priceToBook"),
    #     "Profit Margin": stock.info.get("profitMargins"),
    #     "Diluted EPS": stock.info.get("trailingEps"),
    #     "Beta": stock.info.get("beta")
    # }

    # for key, value in fundamentals.items():
    #     stock_data[key] = value

    return stock_data


def fetch_stock_data_polygon(ticker, api_key, start_date='2015-01-01', end_date='2024-01-01'):
    """
    Fetch historical daily OHLCV stock data from Polygon.io.
    
    Parameters:
        ticker (str): Stock symbol (e.g., "AAPL").
        api_key (str): Your Polygon.io API key.
        start_date (str): Format "YYYY-MM-DD".
        end_date (str): Format "YYYY-MM-DD".
        
    Returns:
        pd.DataFrame: DataFrame with Date index and OHLCV columns.
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Polygon API error: {response.status_code} - {response.text}")
    
    data = response.json()
    if 'results' not in data:
        raise ValueError(f"No data returned for {ticker}. Response: {data}")
    
    df = pd.DataFrame(data['results'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={
        't': 'Date',
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume',
        'vw': 'VWAP',
        'n': 'Trade Count'
    })
    df.set_index('Date', inplace=True)
    return df

def fetch_stock_data_alpha(ticker, api_key='NL9PDOM5JWRPAT9O', start_date='2010-01-01', end_date=None):
    """
    Fetch daily OHLCV data + key fundamentals from Alpha Vantage.
    """
    # --- Fetch OHLCV data ---
    url_price = 'https://www.alphavantage.co/query'
    price_params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key
    }

    r_price = requests.get(url_price, params=price_params)
    price_data = r_price.json()

    if "Time Series (Daily)" not in price_data:
        raise Exception(f"Alpha Vantage price error: {price_data.get('Note') or price_data}")

    df = pd.DataFrame.from_dict(price_data["Time Series (Daily)"], orient="index", dtype='float')
    df.index = pd.to_datetime(df.index)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.sort_index()
    if end_date:
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    else:
        df = df[df.index >= start_date]

    # --- Fetch fundamentals ---
    url_fund = 'https://www.alphavantage.co/query'
    fund_params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": api_key
    }

    r_fund = requests.get(url_fund, params=fund_params)
    fund_data = r_fund.json()

    if not fund_data or "Symbol" not in fund_data:
        raise Exception(f"Alpha Vantage fundamentals error: {fund_data.get('Note') or fund_data}")

    fundamentals = {
        "DE Ratio": float(fund_data.get("DebtEquity", "nan")),
        "Return on Equity": float(fund_data.get("ReturnOnEquityTTM", "nan")),
        "Price/Book": float(fund_data.get("PriceToBookRatio", "nan")),
        "Profit Margin": float(fund_data.get("ProfitMargin", "nan")),
        "Diluted EPS": float(fund_data.get("EPS", "nan")),
        "Beta": float(fund_data.get("Beta", "nan")),
    }

    # --- Add fundamentals to each row in the OHLCV DataFrame ---
    for key, value in fundamentals.items():
        df[key] = value

    return df

def save_data_to_csv(df, filename='tsla_data.csv'):
    """
    Save the stock data to a CSV file.
    
    Args:
    df: DataFrame, The data to save.
    filename: str, The name of the file to save the data to.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename)
    print(f"Data saved to {filename}")

def load_data_from_csv(filename='tsla_data.csv'):
    """
    Load stock data from a CSV file.
    
    Args:
    filename: str, The name of the CSV file.
    
    Returns:
    df: DataFrame, The loaded data.
    """
    return pd.read_csv(filename)

def create_lstm_input(data, target_column='close', lookback=20):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data.iloc[i-lookback:i].values)
        y.append(data.iloc[i][target_column]) 
    return np.array(X), np.array(y)

def preprocess_data(data):
    """
    Preprocess the stock data by handling missing values and scaling the data.
    
    Args:
    df: DataFrame, The raw stock data.
    
    Returns:
    df: DataFrame, The preprocessed data.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Adj Close', 'Volume', 'DE Ratio', 'Return on Equity', 
                                             'Price/Book', 'Profit Margin', 'Diluted EPS', 'Beta']])
    return pd.DataFrame(scaled_data, columns=['Adj Close', 'Volume', 'DE Ratio', 'Return on Equity', 
                                              'Price/Book', 'Profit Margin', 'Diluted EPS', 'Beta']), scaler


def preprocess_data_alpha(data):
    """
    Preprocess the stock data by handling missing values and scaling the data.
    
    Args:
        data (pd.DataFrame): The raw stock data with OHLCV and fundamental columns.
    
    Returns:
        pd.DataFrame: Scaled feature DataFrame.
        MinMaxScaler: Fitted scaler for inverse transform or later use.
    """
    features = ['Close', 'Volume', 'DE Ratio', 'Return on Equity', 
                'Price/Book', 'Profit Margin', 'Diluted EPS', 'Beta']
    
    # If any of these columns are missing, raise a clearer error
    missing_columns = [f for f in features if f not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Fill NaNs in fundamental fields with their column mean (or zero)
    data[features] = data[features].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Drop any remaining NaNs (just in case)
    df_clean = data[features].dropna()
    if df_clean.empty:
        raise ValueError("No valid data found after cleaning. Check for fundamental values being all NaN.")

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_clean)

    return pd.DataFrame(scaled_data, columns=features, index=df_clean.index), scaler


def fetch_data_up_to_last_week(ticker='TSLA', start_date='2023-01-01', end_date='2024-09-27'):
    """
    Fetch stock data up to the date just before last week.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    ticker_info = yf.Ticker(ticker)
    fundamentals = {
        'DE Ratio': ticker_info.info.get('debtToEquity', None),
        'Return on Equity': ticker_info.info.get('returnOnEquity', None),
        'Price/Book': ticker_info.info.get('priceToBook', None),
        'Profit Margin': ticker_info.info.get('profitMargins', None),
        'Diluted EPS': ticker_info.info.get('trailingEps', None),
        'Beta': ticker_info.info.get('beta', None)
    }
    
    fundamentals_df = pd.DataFrame([fundamentals] * len(stock_data), index=stock_data.index)
    
    combined_data = pd.concat([stock_data, fundamentals_df], axis=1)
    
    return combined_data

def fetch_fundamentals_alpha(ticker, api_key='BTR4OON08VH41NYX'):
    """
    Fetch key financial ratios for a stock using Alpha Vantage's OVERVIEW endpoint.
    """
    url = 'https://www.alphavantage.co/query'
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": api_key
    }

    r = requests.get(url, params=params)
    data = r.json()

    if not data or "Symbol" not in data:
        raise Exception(f"Failed to fetch fundamentals for {ticker}: {data.get('Note') or data}")

    fundamentals = {
        "DE Ratio": float(data.get("DebtEquity", "nan")),
        "Return on Equity": float(data.get("ReturnOnEquityTTM", "nan")),
        "Price/Book": float(data.get("PriceToBookRatio", "nan")),
        "Profit Margin": float(data.get("ProfitMargin", "nan")),
        "Diluted EPS": float(data.get("EPS", "nan")),
        "Beta": float(data.get("Beta", "nan")),
    }

    return fundamentals

def fetch_last_week_data(ticker='TSLA', start_date='2024-09-30', end_date='2024-10-05'):
    """
    Fetch stock data for last week.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def fetch_last_month_data(ticker='TSLA', start_date='2024-09-01', end_date='2024-10-01'):
    """
    Fetch stock data for last week.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def fetch_data_prior_to_last_month(ticker='TSLA', start_date='2008-01-01', last_month_start='2024-09-01'):
    """
    Fetch stock data up to the start of the last month and include fundamental indicators.
    """
    # Download historical stock price data
    stock_data = yf.download(ticker, start=start_date, end=last_month_start)
    
    # Fetch fundamental data
    ticker_info = yf.Ticker(ticker)
    fundamentals = {
        'DE Ratio': ticker_info.info.get('debtToEquity', None),
        'Return on Equity': ticker_info.info.get('returnOnEquity', None),
        'Price/Book': ticker_info.info.get('priceToBook', None),
        'Profit Margin': ticker_info.info.get('profitMargins', None),
        'Diluted EPS': ticker_info.info.get('trailingEps', None),
        'Beta': ticker_info.info.get('beta', None)
    }
    
    # Create a DataFrame for fundamental data, repeated for each date in stock_data
    fundamentals_df = pd.DataFrame([fundamentals] * len(stock_data), index=stock_data.index)
    
    # Concatenate stock price data with the fundamental indicators
    combined_data = pd.concat([stock_data, fundamentals_df], axis=1)
    
    return combined_data

import talib as ta

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for stock data.
    
    Args:
    df: DataFrame, The stock data.
    
    Returns:
    df: DataFrame, The stock data with added indicators.
    """
    # Historical Prices are already in df (Open, High, Low, Close, Volume)

    # Overlap Studies Indicators
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(df['Close'], timeperiod=20)
    df['DEMA'] = ta.DEMA(df['Close'], timeperiod=30)
    df['MIDPOINT'] = ta.MIDPOINT(df['Close'], timeperiod=14)
    df['MIDPRICE'] = ta.MIDPRICE(df['High'], df['Low'], timeperiod=14)
    df['SMA'] = ta.SMA(df['Close'], timeperiod=30)
    df['T3'] = ta.T3(df['Close'], timeperiod=5, vfactor=0.7)
    df['TEMA'] = ta.TEMA(df['Close'], timeperiod=30)
    df['TRIMA'] = ta.TRIMA(df['Close'], timeperiod=30)
    df['WMA'] = ta.WMA(df['Close'], timeperiod=30)

    # Momentum Indicators
    df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ADXR'] = ta.ADXR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['APO'] = ta.APO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
    df['AROON_DOWN'], df['AROON_UP'] = ta.AROON(df['High'], df['Low'], timeperiod=14)
    df['AROONOSC'] = ta.AROONOSC(df['High'], df['Low'], timeperiod=14)
    df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CMO'] = ta.CMO(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MFI'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    df['MINUS_DI'] = ta.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MINUS_DM'] = ta.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    df['MOM'] = ta.MOM(df['Close'], timeperiod=10)
    df['PLUS_DI'] = ta.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['PLUS_DM'] = ta.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    df['ROC'] = ta.ROC(df['Close'], timeperiod=10)
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    df['STOCH_slowk'], df['STOCH_slowd'] = ta.STOCH(df['High'], df['Low'], df['Close'], 
                                                      fastk_period=5, slowk_period=3, slowk_matype=0, 
                                                      slowd_period=3, slowd_matype=0)
    df['STOCH_fastk'], df['STOCH_fastd'] = ta.STOCHF(df['High'], df['Low'], df['Close'], 
                                                     fastk_period=5, fastd_period=3, fastd_matype=0)
    
    # Volatility Indicators
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['NATR'] = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['TRANGE'] = ta.TRANGE(df['High'], df['Low'], df['Close'])

    # Volume Indicators
    df['AD'] = ta.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    df['ADOSC'] = ta.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])

    # Price Transform Indicators
    df['AVGPRICE'] = ta.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'])
    df['MEDPRICE'] = ta.MEDPRICE(df['High'], df['Low'])
    df['TYPPRICE'] = ta.TYPPRICE(df['High'], df['Low'], df['Close'])
    df['WCLPRICE'] = ta.WCLPRICE(df['High'], df['Low'], df['Close'])

    # Cycle Indicators
    df['HT_DCPERIOD'] = ta.HT_DCPERIOD(df['Close'])
    df['HT_DCPHASE'] = ta.HT_DCPHASE(df['Close'])
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = ta.HT_PHASOR(df['Close'])
    df['HT_SINE'], df['HT_LEADSINE'] = ta.HT_SINE(df['Close'])
    df['HT_TRENDMODE'] = ta.HT_TRENDMODE(df['Close'])
    
    return df


if __name__ == "__name__":
    tickers = ["AMZN", "TSLA"]
    data = pd.DataFrame()
    for ticker in tickers:
        data[ticker] = fetch_stock_data_alpha(ticker, start_date= "2020-01-01", end_date="2025-05-01")
    
    print(data)
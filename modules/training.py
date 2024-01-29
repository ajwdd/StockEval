import logging
import threading
import time
from colorama import Fore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from modules.calcs import calculate_rsi
import yfinance as yf


def get_stock_data(ticker, start_date, end_date):
    """
    Returns stock data for a given ticker symbol and date range.
    _summary_

    Args:
        ticker (_type_): _description_
        start_date (_type_): _description_
        end_date (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except yf.TickerError:
        logging.warning(f"Unknown ticker: {ticker}")
        print(Fore.RED + f"Unknown ticker: {ticker}" + Fore.RESET)
        return None
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        print(Fore.RED + f"Error fetching stock data: {e}" + Fore.RESET)
        return None


def preprocess_data(stock_data):
    try:
        stock_data["Date"] = stock_data.index
        stock_data.reset_index(drop=True, inplace=True)
        return stock_data
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        print(Fore.RED + f"Error preprocessing data: {e}" + Fore.RESET)
        return None


def create_features(stock_data):
    """
    Creates features for the stock data.
    _summary_

    Args:
        stock_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        stock_data["SMA_50"] = stock_data["Close"].rolling(window=50).mean()
        stock_data["SMA_200"] = stock_data["Close"].rolling(window=200).mean()
        stock_data["EMA_12"] = stock_data["Close"].ewm(span=12, adjust=False).mean()
        stock_data["EMA_26"] = stock_data["Close"].ewm(span=26, adjust=False).mean()
        stock_data["MACD"] = stock_data["EMA_12"] - stock_data["EMA_26"]
        stock_data["RSI"] = calculate_rsi(stock_data["Close"], window=14)

        stock_data["Daily_Return"] = stock_data["Close"].pct_change()

        for lag in range(1, 6):
            stock_data[f"Close_Lag_{lag}"] = stock_data["Close"].shift(lag)
            stock_data[f"Daily_Return_Lag_{lag}"] = stock_data["Daily_Return"].shift(
                lag
            )

        stock_data["Rolling_Mean_Close"] = stock_data["Close"].rolling(window=10).mean()
        stock_data["Rolling_Std_Close"] = stock_data["Close"].rolling(window=10).std()

        stock_data["Future_Close"] = stock_data["Close"].shift(-1)

        stock_data = stock_data.dropna()

        return stock_data
    except Exception as e:
        logging.error(f"Error creating features: {e}")
        print(f"Error creating features: {e}")
        return None


def train_model(features, target):
    global training
    training = True

    try:
        if len(features) < 2:
            logging.warning("Not enough samples to train the model.")
            print(
                "\r"
                + Fore.RED
                + "Not enough samples to train the model."
                + Fore.RESET
                + " " * 30
            )
            training = False
            return None, None

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Sequential()
        model.add(
            LSTM(units=250, activation="relu", input_shape=(X_train_scaled.shape[1], 1))
        )
        model.add(Dense(units=1))
        model.compile(optimizer="adam", loss="mse")

        X_train_reshaped = X_train_scaled.reshape(
            (X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        )
        X_test_reshaped = X_test_scaled.reshape(
            (X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        )

        model.fit(
            X_train_reshaped,
            y_train,
            epochs=1000,
            batch_size=128,
            validation_data=(X_test_reshaped, y_test),
            verbose=3,
        )

        return model, scaler
    except Exception as e:
        logging.error(f"Error training model: {e}")
        print(Fore.RED + f"Error training model: {e}" + Fore.RESET)
        return None, None

    finally:
        training = False

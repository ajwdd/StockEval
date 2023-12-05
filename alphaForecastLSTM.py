import datetime
import logging
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig(
    filename="stock-model.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# ANSI escape codes for text color
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

warnings.simplefilter(action="ignore", category=FutureWarning)



def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except yf.TickerError:
        logging.warning(f"Unknown ticker: {ticker}")
        print(f"Unknown ticker: {ticker}")
        return None
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        print(f"Error fetching stock data: {e}")
        return None


def preprocess_data(stock_data):
    try:
        stock_data["Date"] = stock_data.index
        stock_data.reset_index(drop=True, inplace=True)
        return stock_data
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        print(f"Error preprocessing data: {e}")
        return None


def create_features(stock_data):
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


def calculate_rsi(data, window=14):
    try:
        diff = data.diff(1)
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        print(f"Error calculating RSI: {e}")
        return None


def predict_future_prices_lstm(model, scaler, current_features, selected_features):
    try:
        # Select the relevant features
        current_features = current_features[selected_features]

        # Scale and reshape the input features
        current_features_scaled = scaler.transform(current_features)
        current_features_reshaped = current_features_scaled.reshape(
            (1, len(selected_features), 1)
        )

        # Make predictions with the LSTM model
        predicted_price = model.predict(current_features_reshaped)

        return predicted_price[0, 0]

    except Exception as e:
        logging.error(f"Error predicting future prices with LSTM: {e}")
        print(f"Error predicting future prices with LSTM: {e}")
        return None


def train_model(features, target):
    try:
        if len(features) < 2:
            logging.warning("Not enough samples to train the model.")
            print("Not enough samples to train the model.")
            return None, None

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Sequential()
        model.add(
            LSTM(units=50, activation="relu", input_shape=(X_train_scaled.shape[1], 1))
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
            epochs=50,
            batch_size=32,
            validation_data=(X_test_reshaped, y_test),
            verbose=1,
        )

        return model, scaler

    except Exception as e:
        logging.error(f"Error training the LSTM model: {e}")
        print(f"Error training the LSTM model: {e}")
        return None, None


def main():
    while True:
        try:
            logging.info("---- Starting a new run ----")
            print("Welcome to Alpha Forecast with LSTM!")

            ticker = input("Enter the stock ticker symbol: ")
            logging.info(f"User entered stock ticker: {ticker}")

            start_date = input("Enter the start date (YYYY-MM-DD): ")
            end_date = input(
                "Enter the end date (YYYY-MM-DD) or specify the number of days (e.g., +30): "
            )

            if "+" in end_date:
                days_to_add = int(end_date[1:])
                end_date = (
                    datetime.datetime.strptime(start_date, "%Y-%m-%d")
                    + datetime.timedelta(days=days_to_add)
                ).strftime("%Y-%m-%d")

            logging.info(f"User entered date range: {start_date} to {end_date}")

            stock_data = get_stock_data(ticker, start_date, end_date)
            if stock_data is None:
                logging.warning(f"Unknown ticker: {ticker}")
                print("Unknown ticker. Please enter a valid stock ticker.")
                retry = input(
                    "Do you want to re-enter the information? (yes/no): "
                ).lower()
                if retry != "yes":
                    break

                continue

            stock_data = preprocess_data(stock_data)
            if stock_data is None:
                logging.warning("Error preprocessing data")
                print("Error preprocessing data. Please re-enter the information.")
                retry = input(
                    "Do you want to re-enter the information? (yes/no): "
                ).lower()
                if retry != "yes":
                    break

                continue

            features = create_features(stock_data)
            if features is None:
                logging.warning("Error creating features")
                print("Error creating features. Please re-enter the information.")
                retry = input(
                    "Do you want to re-enter the information? (yes/no): "
                ).lower()
                if retry != "yes":
                    break

                continue

            selected_features = [
                "Open",
                "High",
                "Low",
                "Volume",
                "SMA_50",
                "SMA_200",
                "MACD",
                "RSI",
            ]
            X = features[selected_features]
            y = features["Future_Close"]

            lstm_model, scaler = train_model(X, y)
            if lstm_model is not None and scaler is not None:
                future_date = datetime.datetime.now() + datetime.timedelta(days=1)
                future_features = pd.DataFrame(
                    [
                        [
                            stock_data["Open"].iloc[-1],
                            stock_data["High"].iloc[-1],
                            stock_data["Low"].iloc[-1],
                            stock_data["Volume"].iloc[-1],
                            stock_data["SMA_50"].iloc[-1],
                            stock_data["SMA_200"].iloc[-1],
                            stock_data["MACD"].iloc[-1],
                            stock_data["RSI"].iloc[-1],
                        ]
                    ],
                    columns=selected_features,
                )

                future_price_lstm = predict_future_prices_lstm(
                    lstm_model, scaler, future_features, selected_features
                )


            if future_price_lstm is not None:
                last_close = stock_data["Close"].iloc[-1]
                change_percentage = ((future_price_lstm - last_close) / last_close) * 100

                if future_price_lstm > last_close:
                    direction = "up"
                    color = GREEN  # set text color to green for "up"
                else:
                    direction = "down"
                    color = RED  # set text color to red for "down"

                # Print predicted close price with color-coded direction
                print(
                    f"Predicted Close Price with LSTM for {future_date.date()}: {color}{future_price_lstm:.2f}{RESET}"
                )
                print(
                    f"The price is expected to move {direction} by {color}{abs(change_percentage):.2f}%{RESET} from the last close."
                )

            logging.info("---- Run completed successfully ----")
            break

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred: {e}")
            retry = input("Do you want to re-enter the information? (yes/no): ").lower()
            if retry != "yes":
                break

if __name__ == "__main__":
    main()

import datetime
import logging
import datetime
import logging
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Define ANSI escape codes for text formatting
GREEN = '\033[92m'  # Green color
RED = '\033[91m'    # Red color
RESET = '\033[0m'   # Reset to default color

def print_colored_text(text, color):
    print(f'{color}{text}{RESET}')

logging.basicConfig(
    filename="stock-model.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


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
        # Add your existing feature creation code here
        stock_data["SMA_50"] = stock_data["Close"].rolling(window=50).mean()
        stock_data["SMA_200"] = stock_data["Close"].rolling(window=200).mean()
        stock_data["EMA_12"] = stock_data["Close"].ewm(span=12, adjust=False).mean()
        stock_data["EMA_26"] = stock_data["Close"].ewm(span=26, adjust=False).mean()
        stock_data["MACD"] = stock_data["EMA_12"] - stock_data["EMA_26"]
        stock_data["RSI"] = calculate_rsi(stock_data["Close"], window=14)

        # Daily returns
        stock_data["Daily_Return"] = stock_data["Close"].pct_change()

        # Lagged features
        for lag in range(1, 6):
            stock_data[f"Close_Lag_{lag}"] = stock_data["Close"].shift(lag)
            stock_data[f"Daily_Return_Lag_{lag}"] = stock_data["Daily_Return"].shift(
                lag
            )

        # Advanced time series features
        stock_data["Rolling_Mean_Close"] = stock_data["Close"].rolling(window=10).mean()
        stock_data["Rolling_Std_Close"] = stock_data["Close"].rolling(window=10).std()

        # Shift Close to represent Future_Close
        stock_data["Future_Close"] = stock_data["Close"].shift(-1)

        # Drop NaN values introduced by shifting
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


def train_model(features, target):
    try:
        if len(features) < 2:
            logging.warning("Not enough samples to train the model.")
            print("Not enough samples to train the model.")
            return None, None

        # Check if features or target contain sparse data
# Check if features or target contain sparse data
# Check if features or target contain sparse data
        if isinstance(features, pd.DataFrame) and features.dtypes.apply(lambda x: isinstance(x, pd.SparseDtype)).any():
            features = features.to_dense()
        if isinstance(target, pd.Series) and isinstance(target.dtype, pd.SparseDtype):
            target = target.to_dense()

 


        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBRegressor(
            n_estimators=5500,
            learning_rate=0.01,
            max_depth=10,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=50,
        )

        eval_set = [(X_test_scaled, y_test)]
        model.fit(X_train_scaled, y_train, eval_set=eval_set)

        # Evaluate the model
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Mean Squared Error on Test Set: {mse}")
        print(f"Mean Squared Error on Test Set: {mse}")

        return model, scaler
    except Exception as e:
        logging.error(f"Error training the model: {e}")
        print(f"Error training the model: {e}")
        return None, None


def predict_future_prices(model, scaler, current_features):
    try:
        # Assume 'current_features' is a DataFrame with the same features used for training
        current_features_scaled = scaler.transform(current_features)
        predicted_price = model.predict(current_features_scaled)
        return predicted_price
    except Exception as e:
        logging.error(f"Error predicting future prices: {e}")
        print(f"Error predicting future prices: {e}")
        return None


def main():
    while True:
        try:
            logging.info("---- Starting a new run ----")
            print("Welcome to the Alpa Forecast Tool!")

            ticker = input("Enter the stock ticker symbol: ")
            logging.info(f"User entered stock ticker: {ticker}")

            start_date = input("Enter the start date (YYYY-MM-DD): ")
            end_date = input(
                "Enter the end date (YYYY-MM-DD) or specify the number of days (e.g., +30): "
            )

            # Parse the end_date based on user input
            if "+" in end_date:
                days_to_add = int(end_date[1:])
                end_date = (
                    datetime.datetime.strptime(start_date, "%Y-%m-%d")
                    + datetime.timedelta(days=days_to_add)
                ).strftime("%Y-%m-%d")

            logging.info(f"User entered date range: {start_date} to {end_date}")

            # Check if the ticker is valid
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

            model, scaler = train_model(X, y)
            if model is not None and scaler is not None:
                # Predicting future prices for the next day
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

                future_price = predict_future_prices(model, scaler, future_features)[0]

                # Determine if the price is up or down compared to the last close
                last_close = stock_data["Close"].iloc[-1]
                change_percentage = ((future_price - last_close) / last_close) * 100

                if future_price > last_close:
                    direction = "up"
                    print_colored_text(
                        f"Predicted Close Price for {future_date.date()}: {future_price:.2f}",
                        GREEN
                    )
                    print_colored_text(
                        f"The price is expected to move {direction} by {abs(change_percentage):.2f}% from the last close.",
                        GREEN
                    )
                else:
                    direction = "down"
                    print_colored_text(
                        f"Predicted Close Price for {future_date.date()}: {future_price:.2f}",
                        RED
                    )
                    print_colored_text(
                        f"The price is expected to move {direction} by {abs(change_percentage):.2f}% from the last close.",
                        RED
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
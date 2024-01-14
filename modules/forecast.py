import logging
import datetime
import pandas as pd
from colorama import Fore
from modules.calcs import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_maximum_drawdown,
    interpret_ratio,
    interpret_drawdown,
    risk_free_rate,
    predict_future_prices,
)
from modules.training import (
    train_model,
    get_stock_data,
    preprocess_data,
    create_features,
    create_features,
)


def run_forecast():
    while True:
        try:
            logging.info("---- Starting a new run ----")

            ticker = input("Enter the stock ticker symbol: ")
            logging.info(f"User entered stock ticker: {ticker}")

            start_date = input("Enter the start date (YYYY-MM-DD): ")
            end_date = input("Enter the end date (YYYY-MM-DD): ")

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
                print(
                    Fore.YELLOW
                    + "Unknown ticker. Please enter a valid stock ticker."
                    + Fore.RESET
                )
                retry = input(
                    "Do you want to re-enter the information? (yes/no): "
                ).lower()
                if retry != "yes":
                    break

                continue

            stock_data = preprocess_data(stock_data)
            if stock_data is None:
                logging.warning("Error preprocessing data")
                print(
                    Fore.RED
                    + "Error preprocessing data. Please re-enter the information."
                    + Fore.RESET
                )
                retry = input(
                    "Do you want to re-enter the information? (yes/no): "
                ).lower()
                if retry != "yes":
                    break

                continue
            # Calculate daily returns
            daily_returns = stock_data["Close"].pct_change().dropna()

            # Calculate and interpret metrics
            sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
            sortino_ratio = calculate_sortino_ratio(daily_returns, risk_free_rate)
            max_drawdown = calculate_maximum_drawdown(daily_returns)

            print(interpret_ratio(sharpe_ratio, "Sharpe Ratio"))
            print(interpret_ratio(sortino_ratio, "Sortino Ratio"))
            print(interpret_drawdown(max_drawdown))
            features = create_features(stock_data)
            if features is None:
                logging.warning("Error creating features")
                print(
                    Fore.RED
                    + "Error creating features. Please re-enter the information."
                    + Fore.RESET
                )
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

                future_price_lstm = predict_future_prices(
                    lstm_model, scaler, future_features, selected_features
                )

            if future_price_lstm is not None:
                last_close = stock_data["Close"].iloc[-1]
                change_percentage = (
                    (future_price_lstm - last_close) / last_close
                ) * 100

                if future_price_lstm > last_close:
                    direction = Fore.GREEN + "up" + Fore.RESET
                else:
                    direction = Fore.RED + "down" + Fore.RESET

                # Print predicted close price with color-coded direction
                print(
                    f"Forecasted close price for {future_date.date()}: {future_price_lstm:.2f}"
                )
                print(
                    f"The price is expected to move {direction} by {abs(change_percentage):.2f}% from the last close."
                )

            logging.info("---- Run completed successfully ----")
            break

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred: {e}")
            retry = input("Do you want to re-enter the information? (yes/no): ").lower()
            if retry != "yes":
                break

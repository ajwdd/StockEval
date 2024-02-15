import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from colorama import Fore
import logging
from modules.forecast import run_forecast
from modules.sentiment import run_sentiment
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig(
    filename="stock-model.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)
intro = r"""
┌────────────────────────────────────┐
│        _____                       │
│  __ __/ __(_)__  ___ ____  _______ │
│ / // / _// / _ \/ _ `/ _ \/ __/ -_)│
│ \_,_/_/ /_/_//_/\_,_/_//_/\__/\__/ │
│                                    │
└────────────────────────────────────┘
"""


def main():
    while True:
        print(intro)
        print("1. Forecast Stock Price")
        print("2. Stock Sentiment Analysis")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            run_forecast()
        elif choice == "2":
            run_sentiment()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()

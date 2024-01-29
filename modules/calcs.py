import logging
import numpy as np
import yfinance as yf
from colorama import Fore


# Function to fetch current US 10-Year Treasury yield as the risk-free rate
def get_current_risk_free_rate():
    try:
        treasury_data = yf.Ticker("^TNX")
        hist = treasury_data.history(period="5d")

        # Check if the DataFrame is empty
        if hist.empty:
            logging.error("No data available for the US 10-Year Treasury yield.")
            return None

        last_close = hist["Close"].iloc[-1]
        return last_close / 100
    except Exception as e:
        logging.error(f"Error fetching the risk-free rate: {e}")
        return None


# Sharpe Ratio calculation
def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    return sharpe_ratio * np.sqrt(252)


# Sortino Ratio calculation
def calculate_sortino_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]

    # Check if downside_returns is empty
    if downside_returns.empty:
        return np.nan  # Return NaN if there are no downside returns

    sortino_ratio = np.mean(excess_returns) / np.std(downside_returns, ddof=1)
    return sortino_ratio * np.sqrt(252)


# Maximum Drawdown calculation
def calculate_maximum_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown


# Function to interpret and explain the risk-free rate
def interpret_risk_free_rate(risk_free_rate):
    if risk_free_rate is None:
        return ("Error", Fore.RED + f"The risk-free rate could not be fetched.")
    if risk_free_rate < 0.01:
        return (
            "Low",
            "A low risk-free rate suggests a lower return for safe investments, potentially making riskier assets more attractive.",
        )
    elif risk_free_rate < 0.03:
        return (
            "Moderate",
            "A moderate risk-free rate indicates a balanced return for safe investments.",
        )
    else:
        return (
            "High",
            "A high risk-free rate suggests a higher return for safe investments, which might make riskier assets less attractive.",
        )


# Function to dynamically interpret the Sharpe and Sortino Ratios
def interpret_ratio(ratio, ratio_name):
    if ratio > 2:
        return (
            Fore.GREEN
            + f"Excellent {ratio_name} ({ratio:.2f}){Fore.RESET}: Indicates a very high risk-adjusted return."
        )
    elif ratio > 1:
        return (
            Fore.GREEN
            + f"Good {ratio_name} ({ratio:.2f}){Fore.RESET}: Indicates a high risk-adjusted return."
        )
    elif ratio > 0:
        return (
            Fore.YELLOW
            + f"Fair {ratio_name} ({ratio:.2f}){Fore.RESET}: Indicates some risk-adjusted return, but can be improved."
        )
    else:
        return (
            Fore.RED
            + f"Poor {ratio_name} ({ratio:.2f}){Fore.RESET}: Indicates a negative risk-adjusted return."
        )


# Function to interpret Maximum Drawdown
def interpret_drawdown(drawdown):
    if drawdown > -0.1:
        return (
            Fore.GREEN
            + f"Low risk ({drawdown:.2%} drawdown){Fore.RESET}: Indicates a lower downside risk."
        )
    elif drawdown > -0.2:
        return (
            Fore.YELLOW
            + f"Moderate risk ({drawdown:.2%} drawdown){Fore.RESET}: Indicates a moderate downside risk."
        )
    else:
        return (
            Fore.RED
            + f"High risk ({drawdown:.2%} drawdown){Fore.RESET}: Indicates a high downside risk."
        )


# Function to interpret and explain the Sharpe and Sortino Ratios
risk_free_rate = get_current_risk_free_rate()
rf_rate_interpretation, rf_rate_explanation = interpret_risk_free_rate(risk_free_rate)
if risk_free_rate is None:
    risk_free_rate = 0
if risk_free_rate > 0:
    print(f"Current Market's Risk-Free Rate: {risk_free_rate:.2%}")
    print(f"Interpretation: {rf_rate_interpretation}, {rf_rate_explanation}\n")


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
        print(Fore.RED + f"Error calculating RSI: {e}" + Fore.RESET)
        return None


def predict_future_prices(model, scaler, current_features, selected_features):
    try:
        # Select the relevant features
        current_features = current_features[selected_features]

        # Scale and reshape the input features
        current_features_scaled = scaler.transform(current_features)
        current_features_reshaped = current_features_scaled.reshape(
            (1, len(selected_features), 1)
        )

        predicted_price = model.predict(current_features_reshaped)

        return predicted_price[0, 0]

    except Exception as e:
        logging.error(f"Error predicting future prices: {e}")
        print(Fore.RED + f"Error predicting future prices: {e}" + Fore.RESET)
        return None

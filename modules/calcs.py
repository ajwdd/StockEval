from colorama import Fore
import logging
import numpy as np
import yfinance as yf


def get_current_risk_free_rate():
    """Fetches the current risk-free rate from Yahoo Finance by using
    the US 10-Year Treasury yield. The risk-free rate is used to calculate
    the Sharpe and Sortino Ratios, which are measures of risk-adjusted return."""
    try:
        treasury_data = yf.Ticker("^TNX")
        hist = treasury_data.history(period="5d")

        if hist.empty:
            logging.error("No data available for the US 10-Year Treasury yield.")
            return None

        last_close = hist["Close"].iloc[-1]
        return last_close / 100
    except Exception as e:
        logging.error(f"Error fetching the risk-free rate: {e}")
        return None


def interpret_risk_free_rate(risk_free_rate):
    """Interprets the risk-free rate and explains how it affects the Sharpe and Sortino Ratios."""
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


risk_free_rate = get_current_risk_free_rate()
rf_rate_interpretation, rf_rate_explanation = interpret_risk_free_rate(risk_free_rate)
if risk_free_rate is None:
    risk_free_rate = 0
if risk_free_rate > 0:
    print(f"Current Market's Risk-Free Rate: {risk_free_rate:.2%}")
    print(f"Interpretation: {rf_rate_interpretation}, {rf_rate_explanation}")


def calculate_sharpe_ratio(returns, risk_free_rate):
    """Calculates the Sharpe Ratio given a set of returns and a risk-free rate.
    The Sharpe Ratio is a measure of risk-adjusted return, and is one of the most popular
    metrics for comparing the performance of investment strategies."""
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    return sharpe_ratio * np.sqrt(252)


def calculate_sortino_ratio(returns, risk_free_rate):
    """Calculates the Sortino Ratio given a set of returns and a risk-free rate.
    The Sortino Ratio is a variation of the Sharpe Ratio that only considers the downside risk,
    and is therefore more suitable for investors with a low risk tolerance."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]

    if downside_returns.empty:
        return np.nan

    sortino_ratio = np.mean(excess_returns) / np.std(downside_returns, ddof=1)
    return sortino_ratio * np.sqrt(252)


def calculate_maximum_drawdown(returns):
    """Calculates the Maximum Drawdown given a set of returns.
    The Maximum Drawdown is a measure of downside risk, and is the largest
    percentage drop in the value of a portfolio from a previous peak."""
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown


def interpret_ratio(ratio, ratio_name):
    """Interprets the Sharpe and Sortino Ratios and explains what they mean."""
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


def interpret_drawdown(drawdown):
    """Interprets the Maximum Drawdown and explains what it means."""
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


def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI) given a set of data and a window.
    The RSI is a momentum indicator that measures the magnitude of recent price changes
    to evaluate overbought or oversold conditions in the price of a stock or other asset.
    """
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
    """Predicts future prices given a model, a scaler, a set of current features, and a set of selected features."""
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

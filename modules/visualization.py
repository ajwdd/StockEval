# visualization.py
import numpy as np
import logging
import pandas as pd
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots

pio.templates.default = "ggplot2"


def visualize_data(stock_symbol, news_data, stock_data):
    """Creates responsive and fluid visualizations for stock prices and news sentiment."""
    # Create DataFrame from news_data
    df = pd.DataFrame(news_data)

    # Check if DataFrame has sentiment data
    if "sentiment" not in df.columns or "source" not in df.columns or df.empty:
        print("Required data (sentiment or URL) is missing.")
        return

    if "source" not in df.columns:
        df["source"] = "N/A"
        print("No URL available for" + df["title"] + ".")
    # Calculate the average sentiment score
    avg_sentiment = df["sentiment"].mean()
    print(f"Average Sentiment Score: {avg_sentiment:.2f}")

    # Fetch historical stock data
    hist = stock_data.history(period="1mo")

    # Creating subplots with adjusted row heights
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{stock_symbol} Stock Price (1 Month)",
            "News Sentiment Heatmap",
        ),
        vertical_spacing=0.15,  # Adjust vertical spacing
        row_heights=[0.5, 0.5],  # Equal row heights to avoid overlapping
    )

    # Adding the stock price line chart
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name=f"{stock_symbol} Data",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(xaxis_rangeslider_visible=False)

    # Shortening titles and sources
    shortened_titles = [
        title[:20] + "..." if len(title) > 20 else title for title in df["title"]
    ]
    shortened_sources = [
        source[:20] + "..." if len(source) > 20 else source for source in df["source"]
    ]

    # Adding the news sentiment heatmap
    fig.add_trace(
        go.Heatmap(
            z=df["sentiment"],
            x=shortened_sources,  # Assuming this is correctly capturing your 'source' data
            y=shortened_titles,
            colorscale="RdBu",
            hoverinfo="none",  # Disabling default hover info
            customdata=np.stack((df["title"], df["source"]), axis=-1),
            hovertemplate="<b>Source: %{customdata[1]}</b><br>Title: %{customdata[0]}<br>Sentiment: %{z}<extra></extra><br>",
        ),
        row=2,
        col=1,
    )
    # Add go.
    # Updating layout for responsive design and margin adjustment
    fig.update_layout(
        autosize=True,
        width=None,  # Set to a specific value if autosize doesn't work as expected
        height=800,  # Adjust based on the need to fit content without scrolling
        showlegend=True,
        hovermode="closest",
        title_text=f"Sentiment Visualization for {stock_symbol} - Avg Sentiment: {avg_sentiment:.2f}",
        title_x=0.5,
        margin=dict(l=5, r=5, t=50, b=20),  # Left, Right, Top, Bottom margins
        legend=dict(orientation="h", yanchor="auto", y=1.02, xanchor="right", x=1),
    )

    fig.show()
    logging.info("Sentiment data visualization completed.")

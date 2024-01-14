# visualization.py
import logging
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def visualize_data(stock_symbol, news_data, stock_data):
    """Create responsive and fluid visualizations for stock prices and news sentiment."""
    # Create DataFrame from news_data
    df = pd.DataFrame(news_data)

    # Check if DataFrame has sentiment data
    if "sentiment" not in df.columns or df.empty:
        print("No sentiment data available.")
        return

    # Calculate the average sentiment score
    avg_sentiment = df["sentiment"].mean()
    print(f"\nAverage Sentiment Score: {avg_sentiment:.2f}")

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
        go.Scatter(
            x=hist.index,
            y=hist["Close"],
            mode="lines+markers",
            name="Stock Price",
            line=dict(width=2, color="blue"),
            marker=dict(size=7, color="red"),
            hoverinfo="x+y",
        ),
        row=1,
        col=1,
    )

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
            x=shortened_sources,
            y=shortened_titles,
            colorscale="RdBu",
            hoverinfo="x+y+z",
            text=df["title"],  # Show full title on hover
        ),
        row=2,
        col=1,
    )

    # Updating layout for responsive design and margin adjustment
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode="closest",
        title_text=f"Sentiment Visualization for {stock_symbol} - Avg Sentiment: {avg_sentiment:.2f}",
        title_x=0.5,
        autosize=True,
        legend=dict(
            orientation="h", yanchor="auto", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    fig.show()
    logging.info("Sentiment data visualization completed.")

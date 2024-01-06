import yfinance as yf
import asyncio
import aiohttp
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize and download necessary resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Implementing caching for fetched news and sentiment analysis
news_cache = {}  # Cache for news articles
sentiment_cache = {}  # Cache for sentiment scores

async def fetch_feed(url, session):
    """Asynchronously fetch a single feed and return its entries."""
    try:
        async with session.get(url) as response:
            text = await response.text()
            feed = feedparser.parse(text)
            return feed.entries
    except Exception as e:
        logging.error(f"Error fetching news from {url}: {e}")
        return []

async def fetch_news(rss_urls, stock_symbol, company_name, target_count):
    """Asynchronously fetch news articles from RSS feeds and cache the results."""
    global news_cache
    cache_key = (stock_symbol, company_name, target_count)
    if cache_key in news_cache:
        logging.info(f"Returning cached news for {stock_symbol} - {company_name}")
        return news_cache[cache_key]

    news_items = []
    fetched_count = 0
    relevant_count = 0

    logging.info(f"Fetching news for {stock_symbol} - {company_name}")

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_feed(url, session) for url in rss_urls]
        feeds = await asyncio.gather(*tasks)

    for entries in feeds:
        if relevant_count >= target_count:
            break
        for entry in entries:
            fetched_count += 1
            if stock_symbol.lower() in entry.title.lower() or any(word.lower() in entry.title.lower() for word in company_name.split()):
                relevant_count += 1
                news_items.append((entry.title, entry.link))
                logging.info(f"Relevant Article #{relevant_count} Found: {entry.title[:50]}")
                if relevant_count >= target_count:
                    break

    logging.info(f"Total Articles Fetched: {fetched_count}, Relevant Articles: {relevant_count}")

    news_cache[cache_key] = news_items
    return news_items

def analyze_sentiment(text):
    """Analyze and cache the sentiment of the given text."""
    global sentiment_cache
    if text in sentiment_cache:
        logging.info("Returning cached sentiment")
        return sentiment_cache[text]  # Return cached sentiment if available

    score = sia.polarity_scores(text)['compound']
    sentiment_cache[text] = score  # Cache the sentiment score
    return score

def visualize_data(stock_symbol, news_data, stock_data):
    """Create visualizations for stock prices and news sentiment."""
    df = pd.DataFrame(news_data)
    hist = stock_data.history(period="1mo")

    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        f"{stock_symbol} Stock Price (1 Month)", "News Sentiment Heatmap"))

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['Close'], mode='lines', name='Stock Price'), row=1, col=1)

    if 'sentiment' in df.columns:
        fig.add_trace(go.Heatmap(
            z=df['sentiment'], x=df['source'], y=df['title'], colorscale='RdBu'), row=2, col=1)

    fig.update_layout(height=800, showlegend=False)
    fig.show()
    logging.info("Data visualization completed")


async def main():
    rss_urls = [
        "https://finance.yahoo.com/rss/topstories",
        "https://fortune.com/feed/",
        "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline",
        "https://www.forbes.com/real-time/feed2/",
        "https://www.marketwatch.com/rss/",
        "http://feeds.marketwatch.com/marketwatch/topstories",
        "http://feeds.marketwatch.com/marketwatch/marketpulse",
        "http://feeds.marketwatch.com/marketwatch/bulletins",
        "http://feeds.marketwatch.com/marketwatch/realtimeheadlines",
        "https://www.huffpost.com/section/business/feed",
        "http://rss.cnn.com/rss/money_news_economy.rss",
        "http://rss.cnn.com/rss/money_news_companies.rss",
        "http://rss.cnn.com/rss/money_latest.rss",
        "http://rss.cnn.com/rss/money_markets.rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.cnbc.com/id/15839069/device/rss/rss.html",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "https://www.cnbc.com/id/10000115/device/rss/rss.html",
        "https://www.ft.com/?format=rss",
        "http://feeds.foxbusiness.com/foxbusiness/latest",
        "http://feeds.foxbusiness.com/foxbusiness/markets",
        "http://feeds.foxbusiness.com/foxbusiness/industries",
        "https://www.bloomberg.com/feed/podcast/etf-report",
        "https://www.bloomberg.com/feed/podcast/masters-in-business",
    ]
    logging.info("Starting main application")
    stock_symbol = input("Enter the stock ticker: ").strip().upper()
    target_article_count = int(
        input("Enter desired number of relevant articles: ").strip())
    stock_info = yf.Ticker(stock_symbol).info
    company_name = stock_info.get('longName', '')

    news_items = await fetch_news(rss_urls, stock_symbol, company_name, target_article_count)

    data = []
    for title, source in news_items:
        sentiment_score = analyze_sentiment(title)
        data.append(
            {'title': title, 'sentiment': sentiment_score, 'source': source})

    if len(data) > 0:
        avg_sentiment = sum(d['sentiment'] for d in data) / len(data)
        print(
            f"\nAverage Sentiment Score for the relevant news articles: {avg_sentiment:.2f}")
    else:
        print("Sentiment data is not available.")

    stock_data = yf.Ticker(stock_symbol)
    visualize_data(stock_symbol, data, stock_data)
    logging.info("Stock data and sentiment analysis completed")
if __name__ == "__main__":
    logging.info("Application started")
    asyncio.run(main())
    logging.info("Application finished")
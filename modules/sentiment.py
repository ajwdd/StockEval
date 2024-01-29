import logging
import aiohttp
import feedparser


from modules.utils import *
from modules.visualization import *
import asyncio
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf

# Implementing caching for fetched news and sentiment analysis
news_cache = {}  # Cache for news articles
sentiment_cache = {}  # Cache for sentiment scores
CACHE_DURATION = timedelta(hours=1)


def is_cache_valid(cache_entry):
    current_time = datetime.now()
    return current_time - cache_entry["timestamp"] < CACHE_DURATION


def clean_up_cache():
    global sentiment_cache
    current_time = datetime.now()
    expired_keys = [
        key
        for key, value in sentiment_cache.items()
        if current_time - value["timestamp"] > CACHE_DURATION
    ]
    for key in expired_keys:
        del sentiment_cache[key]
    logging.info(f"Cleaned up {len(expired_keys)} expired cache entries")


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


async def fetch_feeds(rss_urls, session):
    """Asynchronously fetch multiple RSS feeds."""
    tasks = [fetch_feed(url, session) for url in rss_urls]
    feeds = await asyncio.gather(*tasks)
    return feeds


def filter_relevant_articles(entries, stock_symbol, company_name):
    """
    Filter relevant news articles from the entries.
    _summary_

    Args:
        entries (_type_): _description_
        stock_symbol (_type_): _description_
        company_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    relevant_articles = []
    for entry in entries:
        if stock_symbol.lower() in entry.title.lower() or any(
            word.lower() in entry.title.lower() for word in company_name.split()
        ):
            relevant_articles.append((entry.title, entry.link))
    return relevant_articles


def validate_feed_data(feed_entries):
    """
    Validate the format of the feed entries.
    _summary_

    Args:
        feed_entries (_type_): _description_

    Returns:
        _type_: _description_
    """
    valid_entries = []
    for entry in feed_entries:
        if "title" in entry and "link" in entry:
            valid_entries.append(entry)
        else:
            logging.warning(f"Invalid entry format: {entry}")
    return valid_entries


def log_article_status(fetched_count, relevant_count):
    """Log the status of fetched and relevant articles."""
    logging.info(
        f"Total Articles Fetched: {fetched_count}, Relevant Articles: {relevant_count}"
    )
    print(
        f"Total Articles Fetched: {fetched_count}, Relevant Articles: {relevant_count}"
    )


async def fetch_news(rss_urls, stock_symbol, company_name, target_count):
    """
    Fetch news articles, filter them, and log their status.
    _summary_

    Args:
        rss_urls (_type_): _description_
        stock_symbol (_type_): _description_
        company_name (_type_): _description_
        target_count (_type_): _description_

    Returns:
        _type_: _description_
    """
    global news_cache
    cache_key = (stock_symbol, company_name, target_count)
    if cache_key in news_cache:
        logging.info(f"Returning cached news for {stock_symbol} - {company_name}")
        return news_cache[cache_key]

    news_items = []
    fetched_count = 0
    relevant_count = 0

    async with aiohttp.ClientSession() as session:
        feeds = await fetch_feeds(rss_urls, session)

    for entries in feeds:
        validated_entries = validate_feed_data(entries)
        filtered_articles = filter_relevant_articles(
            validated_entries, stock_symbol, company_name
        )

        for article in filtered_articles:
            if relevant_count < target_count:
                fetched_count += 1
                relevant_count += 1
                news_items.append(article)
                logging.info(
                    f"Relevant Article #{relevant_count} Found: {article[0][:50]}"
                )
                print(f"Relevant Article #{relevant_count} Found: {article[0][:50]}")

    if relevant_count < target_count:
        message = f"Could only find {relevant_count} relevant articles out of the requested {target_count}"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        print(YELLOW + message + RESET)  # Print in yellow
        logging.info(message)

    log_article_status(fetched_count, relevant_count)

    news_cache[cache_key] = news_items
    return news_items


# Initialize and download necessary resources
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()


async def analyze_sentiment(text):
    global sentiment_cache
    current_time = datetime.now()

    # Check cache and expiration
    if text in sentiment_cache and is_cache_valid(sentiment_cache[text]):
        logging.info("Returning cached sentiment")
        return sentiment_cache[text]["score"]

    # Analyze and update cache
    score = sia.polarity_scores(text)["compound"]
    sentiment_cache[text] = {"score": score, "timestamp": current_time}
    return score


async def analyze_sentiment_parallel(texts):
    return await asyncio.gather(*(analyze_sentiment(text) for text in texts))


def run_sentiment():
    verify_feeds = True

    file_path = "config/rss_feeds.json"

    if verify_feeds:
        logging.info("Starting RSS feed verification")
        rss_urls = load_rss_urls(file_path)
        asyncio.run(verify_rss_feeds(rss_urls))

    stock_symbol = input("Enter the stock ticker: ").strip().upper()
    target_article_count = int(
        input("Enter desired number of relevant articles: ").strip()
    )
    stock_info = yf.Ticker(stock_symbol).info
    company_name = stock_info.get("longName", "")
    rss_urls = load_rss_urls(file_path)
    news_items = asyncio.run(
        fetch_news(rss_urls, stock_symbol, company_name, target_article_count)
    )
    sentiments = asyncio.run(
        analyze_sentiment_parallel([article[0] for article in news_items])
    )

    data = [
        {"title": article[0], "sentiment": sentiment, "source": article[1]}
        for article, sentiment in zip(news_items, sentiments)
    ]
    stock_data = yf.Ticker(stock_symbol)
    visualize_data(stock_symbol, data, stock_data)

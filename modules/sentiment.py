import asyncio
import logging
import aiohttp
import feedparser
import nltk
from colorama import Fore
import yfinance as yf
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import string
from modules.config_manager import read_config
from modules.utils import load_rss_urls, verify_rss_feeds
from modules.visualization import visualize_data

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# Caching setup
news_cache = {}
sentiment_cache = {}
CACHE_DURATION = timedelta(hours=1)

sia = SentimentIntensityAnalyzer()


def is_cache_valid(cache_entry):
    """Check if the cache entry is still valid based on the current time."""
    current_time = datetime.now()
    return current_time - cache_entry["timestamp"] < CACHE_DURATION


def clean_up_cache():
    """Remove expired entries from the cache."""
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
    """Filter relevant news articles from the entries."""
    relevant_articles = []
    for entry in entries:
        score = calculate_relevance_score(
            entry.title + " " + entry.get("summary", ""), stock_symbol, company_name
        )
        if score > 0:  # Score threshold can be adjusted
            relevant_articles.append((entry.title, entry.link, score))
    return relevant_articles


def calculate_relevance_score(text, stock_symbol, company_name):
    """Calculate relevance score with advanced criteria including sentiment analysis."""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    filtered_words = [
        word for word in words if word not in stop_words and word.isalpha()
    ]

    score = sum(
        word in filtered_words
        for word in company_name.lower().split() + [stock_symbol.lower()]
    )
    score = adjust_score_with_sentiment(text, score)

    return score


def adjust_score_with_sentiment(text, initial_score):
    """Adjust the relevance score based on sentiment analysis."""
    sentiment_score = sia.polarity_scores(text)["compound"]
    if sentiment_score > 0.5 or sentiment_score < -0.5:
        return initial_score * 1.5
    return initial_score


async def analyze_sentiment(text):
    """Analyze the sentiment of a given text, utilizing caching."""
    global sentiment_cache
    current_time = datetime.now()

    if text in sentiment_cache and is_cache_valid(sentiment_cache[text]):
        logging.info("Returning cached sentiment")
        return sentiment_cache[text]["score"]

    score = sia.polarity_scores(text)["compound"]
    sentiment_cache[text] = {"score": score, "timestamp": current_time}
    return score


async def analyze_sentiment_parallel(texts):
    """Analyze sentiment for multiple texts in parallel."""
    return await asyncio.gather(*(analyze_sentiment(text) for text in texts))


def validate_feed_data(feed_entries):
    """Validate the format of the feed entries."""
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
    """Fetch news articles, filter them, and log their status."""
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
        print(Fore.YELLOW + message + Fore.RESET)
        logging.info(message)

    log_article_status(fetched_count, relevant_count)

    news_cache[cache_key] = news_items
    return news_items


def run_sentiment():
    config = read_config()
    verify_feeds = config.get("verify_rss_on_startup", True)

    file_path = "config/rss_feeds.json"

    if verify_feeds:
        logging.info("Starting RSS feed verification")
        rss_urls = load_rss_urls(file_path)
        asyncio.run(verify_rss_feeds(rss_urls))
    else:
        print(
            Fore.YELLOW
            + "RSS feed verification is disabled in the configuration."
            + Fore.RESET
        )
        logging.info("RSS feed verification is disabled in the configuration.")

    valid_symbol = False
    while not valid_symbol:
        stock_symbol = input("Enter the stock ticker: ").strip().upper()
        try:
            stock_info = yf.Ticker(stock_symbol).info
            if "longName" in stock_info:
                valid_symbol = True
                company_name = stock_info.get("longName", "")
            else:
                print("Invalid stock symbol. Please try again.")
        except Exception as e:
            print(
                Fore.RED
                + f"Error fetching data for symbol {stock_symbol}: {e}. Please try again."
                + Fore.RESET
            )

    valid_article_count = False
    while not valid_article_count:
        try:
            target_article_count_input = input(
                "Enter desired number of relevant articles: "
            ).strip()
            target_article_count = int(target_article_count_input)
            if target_article_count > 0:
                valid_article_count = True
            else:
                print("Please enter a positive integer for the number of articles.")
        except ValueError:
            print(
                "Invalid input. Please enter a valid positive integer for the number of articles."
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

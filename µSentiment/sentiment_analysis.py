# sendiment_analysis.py
import asyncio
from datetime import datetime, timedelta
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

CACHE_DURATION = timedelta(hours=1)

# Implementing caching for fetched news and sentiment analysis
news_cache = {}  # Cache for news articles
sentiment_cache = {}  # Cache for sentiment scores

def is_cache_valid(cache_entry):
    current_time = datetime.now()
    return current_time - cache_entry['timestamp'] < CACHE_DURATION

def clean_up_cache():
    global sentiment_cache
    current_time = datetime.now()
    expired_keys = [key for key, value in sentiment_cache.items() if current_time - value['timestamp'] > CACHE_DURATION]
    for key in expired_keys:
        del sentiment_cache[key]
    logging.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# Initialize and download necessary resources
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

async def analyze_sentiment(text):
    global sentiment_cache
    current_time = datetime.now()

    # Check cache and expiration
    if text in sentiment_cache and is_cache_valid(sentiment_cache[text]):
        logging.info("Returning cached sentiment")
        return sentiment_cache[text]['score']

    # Analyze and update cache
    score = sia.polarity_scores(text)['compound']
    sentiment_cache[text] = {'score': score, 'timestamp': current_time}
    return score

async def analyze_sentiment_parallel(texts):
    return await asyncio.gather(*(analyze_sentiment(text) for text in texts))

# feed_parser.py
import aiohttp
import asyncio
import feedparser
import logging

# Define the news_cache variable here
news_cache = {}

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
        if stock_symbol.lower() in entry.title.lower() or any(
            word.lower() in entry.title.lower() for word in company_name.split()
        ):
            relevant_articles.append((entry.title, entry.link))
    return relevant_articles

def validate_feed_data(feed_entries):
    """Validate the format of the feed entries."""
    valid_entries = []
    for entry in feed_entries:
        if 'title' in entry and 'link' in entry:
            valid_entries.append(entry)
        else:
            logging.warning(f"Invalid entry format: {entry}")
    return valid_entries

def log_article_status(fetched_count, relevant_count):
    """Log the status of fetched and relevant articles."""
    logging.info(
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
        filtered_articles = filter_relevant_articles(validated_entries, stock_symbol, company_name)
        # Rest of the function...
        for article in filtered_articles:
            if relevant_count < target_count:
                fetched_count += 1
                relevant_count += 1
                news_items.append(article)
                logging.info(f"Relevant Article #{relevant_count} Found: {article[0][:50]}")

    if relevant_count < target_count:
        message = f"Could only find {relevant_count} relevant articles out of the requested {target_count}"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        print(YELLOW + message + RESET)  # Print in yellow
        logging.info(message)

    log_article_status(fetched_count, relevant_count)

    news_cache[cache_key] = news_items
    return news_items

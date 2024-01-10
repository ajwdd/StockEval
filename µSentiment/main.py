# main.py
import asyncio
from feed_parser import fetch_news
from sentiment_analysis import analyze_sentiment_parallel
from visualization import visualize_data
from config_manager import read_config, write_config
from utils import verify_rss_feeds, load_rss_urls
import yfinance as yf
import logging
import os


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print("Current working directory:", os.getcwd())

def main():
    verify_feeds = True

    file_path = ("µSentiment/config/rss_feeds.json")
    
    if verify_feeds:
        logging.info("Starting RSS feed verification")
        rss_urls = load_rss_urls(file_path)
        asyncio.run(verify_rss_feeds(rss_urls))
        

    stock_symbol = input("Enter the stock ticker: ").strip().upper()
    target_article_count = int(input("Enter desired number of relevant articles: ").strip())
    stock_info = yf.Ticker(stock_symbol).info
    company_name = stock_info.get("longName", "")
    rss_urls = load_rss_urls(file_path)
    news_items = asyncio.run(fetch_news(rss_urls, stock_symbol, company_name, target_article_count))
    sentiments = asyncio.run(analyze_sentiment_parallel([article[0] for article in news_items]))

    data = [{'title': article[0], 'sentiment': sentiment, 'source': article[1]} for article, sentiment in zip(news_items, sentiments)]
    stock_data = yf.Ticker(stock_symbol)
    visualize_data(stock_symbol, data, stock_data)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
    finally:
        logging.info("Application finished")
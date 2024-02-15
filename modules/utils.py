import aiohttp
import asyncio
from colorama import Fore
import feedparser
import json


async def verify_rss_feed(url, session):
    """Verifies a single RSS feed and return its status."""
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                text = await response.text()
                if feedparser.parse(text).entries:
                    return url, "Accessible and Valid"
                else:
                    return url, "Accessible but Invalid Content"
            else:
                return url, f"Inaccessible, Status Code: {response.status}"
    except asyncio.TimeoutError:
        return url, "Timeout - Feed Not Responding"
    except Exception as e:
        return url, f"Error: {e}"


async def verify_rss_feeds(rss_urls):
    """Verify each RSS feed and log its status with color."""
    async with aiohttp.ClientSession() as session:
        tasks = [verify_rss_feed(url, session) for url in rss_urls]
        results = await asyncio.gather(*tasks)

    if all("Accessible and Valid" in status for url, status in results):
        print(Fore.GREEN + "All RSS feeds accessible and valid" + Fore.RESET)
    else:
        for url, status in results:
            if "Accessible and Valid" not in status:
                print(Fore.RED + f"RSS feed {url} - {status}" + Fore.RESET)


def load_rss_urls(file_path):
    """Load RSS feed URLs from a JSON file."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            # Extract URLs, which are the values of the dictionary
            return list(data.values())
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON.")
        return []

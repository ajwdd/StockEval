import aiohttp
import asyncio
import feedparser
import json

async def verify_rss_feed(url, session):
    """Verify a single RSS feed and return its status."""
    try:
        async with session.get(url, timeout=10) as response:  # 10-second timeout
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

    # ANSI escape codes for colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"  # Reset to default color

    for url, status in results:
        if "Accessible and Valid" in status:
            color = GREEN
        else:
            color = RED
        print(f"{color}RSS feed {url} - {status}{RESET}")

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

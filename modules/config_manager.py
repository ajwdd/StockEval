import json
import os

config_file = "config/config.json"


def ensure_config_exists():
    """Ensure the configuration file exists with default settings."""
    if not os.path.isfile(config_file):
        default_config = {"verify_rss_on_startup": False}
        write_config(default_config)


def read_config():
    """Reads the configuration file and returns the contents."""
    ensure_config_exists()
    try:
        with open(config_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"verify_rss_on_startup": False}


def write_config(config):
    """Write the updated configuration to the file."""
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

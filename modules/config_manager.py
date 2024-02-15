import json
import os

config_file = "config/config.json"  # Adjust the path as necessary

def ensure_config_exists():
    """Ensure the configuration file exists with default settings."""
    if not os.path.isfile(config_file):
        default_config = {"verify_rss_on_startup": False}
        write_config(default_config)

def read_config():
    """Reads the configuration file and returns the contents."""
    ensure_config_exists()  # Ensure the config file exists before reading
    try:
        with open(config_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        # This block should theoretically never be reached due to ensure_config_exists
        return {"verify_rss_on_startup": False}

def write_config(config):
    """Write the updated configuration to the file."""
    os.makedirs(os.path.dirname(config_file), exist_ok=True)  # Ensure directory exists
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

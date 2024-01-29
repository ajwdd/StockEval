import json

config_file = "/config/config.json"


def read_config():
    """
    Reads the configuration file and returns the contents.

    _summary_

    Returns:
        _type_: _description_
    """
    try:
        with open(config_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"verify_rss_on_startup": False}


def write_config(config):
    """Write the updated configuration to the file."""
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

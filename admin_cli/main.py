import argparse
import os
import sys
from pathlib import Path
import logging.config

import requests
from dotenv import load_dotenv

from utils.config_utils import ROOT_DIR, load_yaml_config

LOGGER_CONFIG_PATH = Path(ROOT_DIR / "logger/config")

load_dotenv(Path(ROOT_DIR))

logging.config.dictConfig(
    load_yaml_config(
        f"{LOGGER_CONFIG_PATH}/logger_conf.{os.getenv('ENV', 'testing')}.yaml"
    )
)

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--telegram-id', help='Telegram user ID', type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()

    payload = {
        "telegram_id": args.telegram_id,
    }

    response = requests.post(f"{AUTH_SERVICE_URL}/internal/v1/init_admin", json=payload, timeout=10)

    if response.status_code != 200:
        logging.error(f'Failed to grant admin access! Request ended with status code {response.status_code},'
                      f' and message: {response.text}')

        sys.exit(1)

    logging.info(f'Granted admin permissions to user: {args.telegram_id}!')


if __name__ == "__main__":
    main()

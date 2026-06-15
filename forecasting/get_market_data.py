from __future__ import annotations

import logging
import logging.config
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
LOGGER_CONFIG_PATH = ROOT_DIR / "logger" / "config"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from forecasting.history_parsing.historical_parser import HistoricalParser  # noqa: E402


def _load_yaml_config(path_to_config: Path) -> dict:
    with path_to_config.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logging() -> None:
    env = os.getenv("ENV", "testing")
    config_path = LOGGER_CONFIG_PATH / f"logger_conf.{env}.yaml"
    logs_dir = ROOT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        logging.config.dictConfig(_load_yaml_config(config_path))
        return

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
    logging.warning("logging config not found at %s; using basic logging", config_path)


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    os.environ.setdefault("ENV", "testing")
    setup_logging()

    parser = HistoricalParser()
    parser.convert_data_frame_to_csv(parser.parse_dividends(), output_filename="dividends_data.csv")
    parser.convert_data_frame_to_csv(parser.parse_candles(24), output_filename="historical_data_1d.csv")
    parser.convert_data_frame_to_csv(parser.parse_candles(10), output_filename="historical_data_10min.csv")


if __name__ == "__main__":
    main()

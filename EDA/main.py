import logging.config
import os
from pathlib import Path

from dotenv import load_dotenv

from EDA.history_parsing.historical_parser import HistoricalParser
from utils.config_utils import load_yaml_config, ROOT_DIR

LOGGER_CONFIG_PATH = 'logger/config'


def setup_logging():
    logging.config.dictConfig(load_yaml_config(f'{LOGGER_CONFIG_PATH}/logger_conf.{os.getenv('ENV')}.yaml'))


if __name__ == "__main__":
    load_dotenv(Path(ROOT_DIR / '.env'))
    print(f'{LOGGER_CONFIG_PATH}/logger_conf.{os.getenv('ENV')}.yaml')

    Path(ROOT_DIR / 'logs').mkdir(parents=True, exist_ok=True)

    setup_logging()

    parser = HistoricalParser()
    parser.convert_data_frame_to_csv(parser.parse_dividends(), output_filename='dividends_data.csv')
    parser.convert_data_frame_to_csv(parser.parse_candles(24), output_filename='historical_data_1d.csv')
    parser.convert_data_frame_to_csv(parser.parse_candles(10), output_filename='historical_data_10min.csv')

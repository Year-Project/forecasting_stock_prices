import logging

import requests
from pathlib import Path

import apimoex
import pandas as pd

from utils.utils import ROOT_DIR, load_yaml_config


class HistoricalParser:
    def __init__(self):
        parsed_yaml = load_yaml_config("HISTORY_PARSER_CONF_PATH")
        self.stock_names = parsed_yaml['stocks']
        self.output_dir = parsed_yaml['output_dir']
        self.output_file = parsed_yaml['output_file']

    def parse(self, interval: int = 24) -> pd.DataFrame:
        with requests.Session() as session:
            result = pd.DataFrame()
            for stock_name in self.stock_names:
                logging.info(f'begin data parsing for share "{stock_name}"')

                df = pd.DataFrame(apimoex.get_market_candles(session, stock_name, interval=interval))

                logging.debug(f'saved {df.shape[0]} rows for share "{stock_name}"')

                df.set_index('begin', inplace=True)
                df['name'] = stock_name
                result = pd.concat([result, df])

        return result

    def convert_data_frame_to_csv(self, df: pd.DataFrame, output_filename: str = ""):
        Path(ROOT_DIR / self.output_dir).mkdir(parents=True, exist_ok=True)

        if len(output_filename) == 0:
            output_filename = self.output_file

        destination_path = Path(ROOT_DIR / self.output_dir / output_filename)
        df.to_csv(destination_path, sep=';', encoding="utf-8")

        logging.info(f'csv data exported into file {destination_path}')

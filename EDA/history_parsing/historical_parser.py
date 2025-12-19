import logging
import os

import requests
from pathlib import Path

import apimoex
import pandas as pd
from apimoex.requests import _make_query, _get_short_data

from utils.utils import ROOT_DIR, load_yaml_config


class HistoricalParser:
    CONFIG_PATH = 'EDA/history_parsing/config'

    def __init__(self):
        parsed_yaml = load_yaml_config(f'{self.CONFIG_PATH}/history_parser_conf.{os.getenv('ENV')}.yaml')
        self.stock_names = parsed_yaml['stocks']
        self.output_dir = parsed_yaml['output_dir']
        self.output_file = parsed_yaml['output_file']

    def parse_candles(self, interval: int = 24) -> pd.DataFrame:
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

    def _internal_parse_dividends(self, session: requests.Session, security: str) -> list[dict[str, str | int | float]]:
        url = f"https://iss.moex.com/iss/securities/{security}/dividends.json"
        table = "dividends"
        query = _make_query(q=security, table=table)
        return _get_short_data(session, url, table, query)

    def parse_dividends(self) -> pd.DataFrame:
        with requests.Session() as session:
            result = pd.DataFrame()
            for stock_name in self.stock_names:
                logging.info(f'begin dividend parsing for share "{stock_name}"')

                df = pd.DataFrame(self._internal_parse_dividends(session, stock_name))

                logging.debug(f'saved {df.shape[0]} rows for share "{stock_name}"')

                df.rename(columns={'secid': 'name', 'registryclosedate': 'date'}, inplace=True)
                df.set_index('date', inplace=True)
                result = pd.concat([result, df])

        return result

    def convert_data_frame_to_csv(self, df: pd.DataFrame, output_filename: str = ""):
        Path(ROOT_DIR / self.output_dir).mkdir(parents=True, exist_ok=True)

        if len(output_filename) == 0:
            output_filename = self.output_file

        destination_path = Path(ROOT_DIR / self.output_dir / output_filename)
        df.to_csv(destination_path, sep=';', encoding="utf-8")

        logging.info(f'csv data exported into file {destination_path}')

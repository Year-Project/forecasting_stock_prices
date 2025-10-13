import requests
import os
from dotenv import load_dotenv
from pathlib import Path

import apimoex
import pandas as pd
import yaml


class HistoricalParser:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent

        load_dotenv()
        with open(Path(self.base_dir / os.getenv("STOCKS_HISTORY_PARSER_PATH")), "r", encoding="utf-8") as file:
            parsed_yaml = yaml.safe_load(file)
            self.stock_names = parsed_yaml['stocks']
            self.output_dir = parsed_yaml['output_dir']
            self.output_file = parsed_yaml['output_file']

    def parse(self) -> pd.DataFrame:
        with requests.Session() as session:
            result = pd.DataFrame()
            for stock_name in self.stock_names:
                df = pd.DataFrame(apimoex.get_market_candles(session, stock_name, interval=24))
                df.set_index('begin', inplace=True)
                df['name'] = stock_name
                result = pd.concat([result, df])

        return result

    def convert_data_frame_to_csv(self, df: pd.DataFrame):
        Path(self.base_dir / self.output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(self.base_dir / self.output_dir / self.output_file), sep=';', index=False, encoding="utf-8")

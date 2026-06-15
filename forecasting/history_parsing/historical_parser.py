from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import apimoex
import pandas as pd
import requests
import yaml
from apimoex.requests import _get_short_data, _make_query


ROOT_DIR = Path(__file__).resolve().parents[2]


class HistoricalParser:
    CONFIG_DIR = Path(__file__).resolve().parent / "config"

    def __init__(self, env: str | None = None):
        config_env = env or os.getenv("ENV", "testing")
        config_path = self.CONFIG_DIR / f"history_parser_conf.{config_env}.yaml"
        parsed_yaml = self._load_config(config_path)

        self.stock_names = parsed_yaml["stocks"]
        self.output_dir = Path(parsed_yaml["output_dir"])
        self.output_file = parsed_yaml["output_file"]

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError(f"Expected mapping config in {config_path}")
        return config

    def parse_candles(self, interval: int = 24) -> pd.DataFrame:
        with requests.Session() as session:
            result = pd.DataFrame()
            for stock_name in self.stock_names:
                logging.info('begin data parsing for share "%s"', stock_name)

                df = pd.DataFrame(apimoex.get_market_candles(session, stock_name, interval=interval))
                logging.debug('loaded %s candle rows for share "%s"', df.shape[0], stock_name)

                if df.empty:
                    logging.warning('no candle rows returned for share "%s"', stock_name)
                    continue

                df.set_index("begin", inplace=True)
                df["name"] = stock_name
                result = pd.concat([result, df])

        return result

    def _internal_parse_dividends(
        self, session: requests.Session, security: str
    ) -> list[dict[str, str | int | float]]:
        url = f"https://iss.moex.com/iss/securities/{security}/dividends.json"
        table = "dividends"
        query = _make_query(q=security, table=table)
        return _get_short_data(session, url, table, query)

    def parse_dividends(self) -> pd.DataFrame:
        with requests.Session() as session:
            result = pd.DataFrame()
            for stock_name in self.stock_names:
                logging.info('begin dividend parsing for share "%s"', stock_name)

                df = pd.DataFrame(self._internal_parse_dividends(session, stock_name))
                logging.debug('loaded %s dividend rows for share "%s"', df.shape[0], stock_name)

                if df.empty:
                    logging.warning('no dividend rows returned for share "%s"', stock_name)
                    continue

                df.rename(columns={"secid": "name", "registryclosedate": "date"}, inplace=True)
                df.set_index("date", inplace=True)
                result = pd.concat([result, df])

        return result

    def convert_data_frame_to_csv(self, df: pd.DataFrame, output_filename: str = "") -> Path:
        output_dir = ROOT_DIR / self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(output_filename) == 0:
            output_filename = self.output_file

        destination_path = output_dir / output_filename
        df.to_csv(destination_path, sep=";", encoding="utf-8")

        logging.info("csv data exported into file %s", destination_path)
        return destination_path

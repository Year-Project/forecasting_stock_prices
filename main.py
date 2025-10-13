from history_parsing.historical_parser import HistoricalParser

if __name__ == "__main__":
    parser = HistoricalParser()
    parser.convert_data_frame_to_csv(parser.parse())

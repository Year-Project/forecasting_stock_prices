from EDA.history_parsing.historical_parser import HistoricalParser

if __name__ == "__main__":
    parser = HistoricalParser()
    parser.convert_data_frame_to_csv(parser.parse(24), output_filename='historical_data_1d.csv')
    parser.convert_data_frame_to_csv(parser.parse(10), output_filename='historical_data_10min.csv')

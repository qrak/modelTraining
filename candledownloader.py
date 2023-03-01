import ccxt
import pandas as pd
import time
import os


class CandleDownloader:
    def __init__(self, exchange_name='binance', pair_name='BTC/USDT', timeframe='1h', start_time='2015-01-01T00:00:00Z', end_time=None, batch_size=1000, output_file=None):
        self.exchange = getattr(ccxt, exchange_name)()
        self.pair_name = pair_name
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        self.batch_size = batch_size
        self.output_file = output_file
        self.total_candles = 0
        self.total_batches = 0
        self.buffer = []

        # validate the parameters
        if pair_name not in self.exchange.load_markets():
            raise ValueError(f"Invalid pair name: {pair_name}")
        if timeframe not in self.exchange.timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # set the filename and path for the output file
        if output_file is None:
            symbol_base = pair_name.split('/')[0]
            symbol_quote = pair_name.split('/')[1]
            start_date = start_time.split('T')[0]
            end_date = end_time.split('T')[0] if end_time else 'now'
            filename = f'{symbol_base}_{symbol_quote}_{timeframe}_{start_date}_{end_date}_{self.exchange.id}.csv'
            self.output_file = f'./csv/{filename}'

    def download_candles(self):
        # check if the file already exists
        try:
            df = pd.read_csv(self.output_file, usecols=[0], header=None)
            self.start_time = self.exchange.parse8601(df.iloc[-1, 0]) + (self.exchange.parse_timeframe(self.timeframe) * 1000)
            print(f"Resuming from timestamp {self.start_time}...")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.start_time = self.exchange.parse8601(self.start_time)

        # fetch the candles and write them to the output file
        while True:
            try:
                # fetch the candles for the current time range
                candles = self.exchange.fetch_ohlcv(self.pair_name, self.timeframe, since=self.start_time, limit=self.batch_size)

                if len(candles) == 0:
                    break

                # convert the fetched candles to a pandas dataframe
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # append the dataframe to the in-memory buffer
                self.buffer.append(df)

                # update the start time for the next request
                self.start_time = candles[-1][0] + self.exchange.parse_timeframe(self.timeframe) * 1000

                # check if the end time has been reached
                if self.end_time is not None and self.start_time >= self.exchange.parse8601(self.end_time):
                    break

                # update progress message
                self.total_candles += len(df)
                self.total_batches += 1
                print(f"Downloaded {self.total_candles} candles in {self.total_batches} batches...")

                # write the accumulated data to the output file
                if os.path.isfile(self.output_file):
                    with open(self.output_file, mode='a', newline='') as f:
                        if f.tell() == 0:
                            df.to_csv(f, index=False, header=True)
                        else:
                            df.to_csv(f, index=False, header=False)
                else:
                    df.to_csv(self.output_file, index=False, header=True)

                # clear the buffer
                self.buffer = []

                # pause for a short time to avoid request blocking
                #time.sleep(0.2)

            except Exception as e:
                print(f"Exception occurred: {e}")
                time.sleep(60)

                # write the remaining data in the buffer to the output file
            if len(self.buffer) > 0:
                df = pd.concat(self.buffer)

                if os.path.isfile(self.output_file):
                    with open(self.output_file, mode='a', newline='') as f:
                        if f.tell() == 0:
                            df.to_csv(f, index=False, header=True)
                        else:
                            df.to_csv(f, index=False, header=False)
                else:
                    df.to_csv(self.output_file, index=False, header=True)

                # print a message when the script has finished
            print(
                f'Download complete. Total candles: {self.total_candles}, Total batches: {self.total_batches}, Output file: {self.output_file}')


candledownload = CandleDownloader()
candledownload.download_candles()


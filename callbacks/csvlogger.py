import os
import pandas as pd
from callbacks.basecallback import Callback


class CSVLogger(Callback):

    def __init__(self, csv_path: str, verbose=0):
        super().__init__(name="csvlogger")
        self.csv_path = csv_path
        self.verbose = verbose
        dir_name = os.path.dirname(csv_path)
        if len(dir_name)>0:
            os.makedirs(dir_name, exist_ok=True)
        self.df = pd.DataFrame()

    def on_epoch_end(self, epoch: int, logs: dict):

        epoch_data = pd.DataFrame([logs])

        if self.df.empty:
            self.df = epoch_data
        else:
            self.df = pd.concat([self.df, epoch_data], ignore_index=True)

        self.df.to_csv(self.csv_path, index=False)
        if self.verbose>0:
            print(f"Metrics logged to {self.csv_path} at epoch {epoch}")

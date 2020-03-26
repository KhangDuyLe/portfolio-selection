import pandas as pd 

class ReadData:
    def __init__(self):
        self.name = "read data"

    def read(self, path):
        return pd.read_pickle(path)

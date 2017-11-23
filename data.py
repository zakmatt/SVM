import numpy as np
import pandas as pd


class Data(object):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def get_data(self, col=None, end_col=None, columns=False):
        if col == None and end_col == None:
            return self.data

        if col != None:
            if end_col == None:
                return self.data.iloc[:, col].values
            else:
                if columns:
                    return self.data.iloc[:, [col, end_col]].values
                return self.data.iloc[:, col:end_col + 1].values

        return self.data.iloc[:, :end_col + 1].values

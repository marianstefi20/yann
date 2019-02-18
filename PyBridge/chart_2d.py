import pandas as pd


class Chart2D:
    def __init__(self):
        self.class1 = None
        self.class2 = None

    def read_binary_problem(self, filename):
        pd_data = pd.read_csv('./data/train/linearly_separable.txt', sep=", ")
        self.class1 = pd_data.loc[pd_data['y'] == 1][['x1', 'x2']]
        self.class2 = pd_data.loc[pd_data['y'] == -1][['x1', 'x2']]

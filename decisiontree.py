import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats


def calc_entropy(targets):
    entropy = 0.0
    l = [[x,targets.count(x)] for x in set(targets)]
    for row in l:
            entropy += -row[1]/len(targets) * np.log2(row[1]/len(targets))
    return entropy

def read_file(path):
    path = "C:\\Users\\Jeremy\\Desktop\\" + path
    df = pd.read_csv(path,
                     header=None, na_values="0", delim_whitespace=True)
    a = df.iloc[:, -1:]
    return df.iloc[:, :-1], a


def main():

    x,y = read_file("lenses.data")

    # kf = KFold(n_splits=2)
    # kf.get_n_splits(x)
    # split = kf.split(x,y)
    # main.x_train = None
    # main.x_test = None
    # main.y_train = None
    # main.y_test = None
    # for train_index, test_index in split:
    #     main.x_train, main.x_test = x[train_index], x[test_index]
    #     main.y_train, main.y_test = y[train_index], y[test_index]

    dt = decisionTree(x,y)
    dt.make_tree()


class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


class decisionTree:
    def __init__(self, data_train, targets_train):
        self.data_train = data_train
        self.targets_train = targets_train

    def make_tree(self):
        data = self.data_train
        y = self.targets_train
        root = Tree()
        if len(data) == 0 or len(data.columns) == 0:  # base case 1
            # empty branch
            return y.max()
        elif stats.itemfreq(y.values.flatten()) == len(data):  # base case 2
            return stats.mode(y.values().flatten())
        else:
            classes = []
            averages = []
            for column in data:
                entropies = []
                features = pd.unique(data[column])
                for feature in features:
                    for i in range(len(data[column])):
                        if data.loc[i, column] == feature:
                            classes.append(y.iloc[i, 0])
                    entropies.append(calc_entropy(classes))
                    classes.clear()
                for entropy in entropies:
                    average = 0.0
                    average += entropy * len(features) / len(data[column])
                    averages.append(average)
                    if average < min(averages):
                        root.data = [column, min(averages)]
                entropies.clear()


if __name__ == "__main__":
    main()
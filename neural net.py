import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_iris


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNet:
    def __init__(self, depth, num_inputs, num_nodes):
        self.depth = depth
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.nodes = []
        for x in range(0, depth):
            self.create_layer()

    def generate_weight(self):
        return np.random.uniform(-1, 1)

    def create_layer(self):
        for x in range(0, self.num_nodes):
            weights = []
            for z in range(0, self.num_inputs):
                weights.append(self.generate_weight())
            self.nodes.append(Node(weights))

    def test_inputs(self, inputs):
        for node in self.nodes:
            x = 0
            i = 0
            for weight in node.weights:
                x += weight * inputs[i]
                i += 1
            result = sigmoid(x)[0]
            if result <= 0.5:
                result = 0
            else:
                result = 1
            node.result = result
            print(result)


class Node:
    def __init__(self, weights):
        self.weights = weights
        self.result = 0

def append(x):
    np.append(x, -1)


if __name__ == '__main__':
    iris = load_iris()
    print(iris.data.shape)
    # separate the data from the target attributes
    X = iris.data
    y = iris.target
    normalized_X = preprocessing.normalize(X)
    normalized_X = np.insert(normalized_X, normalized_X.shape[1], -1, axis=1)
    num_cols = normalized_X.shape[1]
    neuralNet = NeuralNet(1, num_cols, 2)
    neuralNet.test_inputs(normalized_X)

    array = np.genfromtxt("/Users/jeremy/Desktop/cs450/pima-indians-diabetes.csv", delimiter=",")
    X = array[:, :-1]
    Y = array[:, -1:]
    normalized_X = preprocessing.normalize(X)
    normalized_X = np.insert(normalized_X, normalized_X.shape[1], -1, axis=1)
    num_cols = normalized_X.shape[1]
    neuralNet = NeuralNet(1, num_cols, 2)
    neuralNet.test_inputs(normalized_X)
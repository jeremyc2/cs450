import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import random


def sigmoid(k):
    return 1 / (1 + np.exp(-k))


class NeuralNet:
    def __init__(self, num_inputs, num_nodes_arr, inputs):
        self.num_inputs = num_inputs
        self.layers = []
        self.targets = []
        self.inputs = inputs
        for k in range(0, len(num_nodes_arr)):
            if k == 0:
                self.layers.append(Layer(num_nodes_arr[k], num_inputs))
            else:
                self.layers.append(Layer(num_nodes_arr[k], self.layers[k - 1].num_nodes))

    def calculate_error_output(self, node, target):
        # print(target)
        error = node.result * (1 - node.result) * (node.result - target)
        return error

    def calculate_error_hidden(self, current_node, next_layer, index_node_current_layer):
        summation = 0
        for node in next_layer.nodes:
            summation += node.error * node.weights[index_node_current_layer]
        error = current_node.result * (1 - current_node.result) * summation
        return error

    def update_weight(self, weight, error, activation_previous_node, j):
        learning_rate = .4
        weight_updated = weight - (learning_rate * error * activation_previous_node)
        # if j == 0:
        #     print("\n Old Weight= " + str(weight) + " New Weight= " + str(weight_updated))
        return weight_updated

    def run(self, target, j):
        inputs = self.inputs
        # For instance in data
        for z in range(0, inputs.shape[0]):
            # Go through each node in each layer
            self.feed_forward(z, j)
            self.backwards_propagation(target, z)
            for l in range(0, len(self.layers)):
                for node_index in range(0, len(self.layers[l].nodes)):
                    for w in range(0, len(self.layers[l].nodes[node_index].weights)):
                        self.layers[l].nodes[node_index].weights[w] = self.update_weight(self.layers[l].nodes[node_index].weights[w], self.layers[l].nodes[node_index].error, self.layers[l].nodes[node_index - 1].result, j)

    def feed_forward(self, z, j):
            inputs = self.inputs
            first = True
            for l in range(0, len(self.layers)):
                for node in self.layers[l].nodes:
                    instance = 0
                    # And calculate it's activation value for each weight
                    for weight in range(0, len(node.weights)):
                        # if first layer use input nodes
                        if l == 0:
                            instance += node.weights[weight] * inputs[z][weight]
                        # else use weights of previous layer
                        else:
                            instance += node.weights[weight] * self.layers[l - 1].nodes[weight].result
                    result = sigmoid(instance)
                    node.result = result
                    if j == 1 and first is True:
                        print(str(node.result) + "= new activation")
                    first = False

    def backwards_propagation(self, target, target_index):
            output_layer = self.layers[-1]
            target_value = target[target_index]
            if target_value == 0:
                target_value = [1, 0, 0]
            elif target_value == 1:
                target_value = [0, 1, 0]
            else:
                target_value = [0, 0, 1]
            # print(target_value)
            index = 0
            for node in output_layer.nodes:
                node.error = self.calculate_error_output(node, target_value[index])
                index += 1
            for layer_index in range(0, len(self.layers) - 1):
                for node_index in range(0, len(self.layers[layer_index].nodes)):
                    node = self.layers[layer_index].nodes[node_index]
                    node.error = self.calculate_error_hidden(node, self.layers[layer_index + 1], node_index)

    def calculate_targets(self):
        inputs = self.inputs
        targets = []
        for z in range(0, inputs.shape[0]):
            result = []
            length = len(self.layers)
            for node in self.layers[length - 1].nodes:
                    if node.result <= 0.5:
                        node.result = 0
                    else:
                        node.result = 1
                    result.append(node.result)
            targets.append(result)
        return targets


class Layer:
    def __init__(self, num_nodes, previous_layer_len, nodes=None):
        self.num_nodes = num_nodes
        self.previous_layer_len = previous_layer_len
        if nodes is not None:
            self.nodes = nodes
        else:
            self.nodes = []
            for j in range(0, self.num_nodes):
                weights = []
                for z in range(0, self.previous_layer_len):
                    weights.append(self.generate_weight())
                self.nodes.append(Node(weights))

    def generate_weight(self):
        weight = random.uniform(-1, 1)
        return weight

    def __str__(self):
        result = []
        for node in self.nodes:
            result.append(node.result)
        return ', '.join(str(j) for j in result)


class Node:
    def __init__(self, weights, error=0):
        self.weights = weights
        self.result = 0
        self.error = error


if __name__ == '__main__':
    iris = load_iris()
    # separate the data from the target attributes
    X = iris.data
    y = iris.target
    # print(y)
    normalized_X = preprocessing.normalize(X)
    normalized_X = np.insert(normalized_X, normalized_X.shape[1], -1, axis=1)
    num_cols = normalized_X.shape[1]
    neuralNet = NeuralNet(num_cols, [5, 7, 2, 3], normalized_X)
    print("\nIris:\n")
    for i in range(0, 200):
        neuralNet.run(y, i)
        results = neuralNet.calculate_targets()
        # print(results)
        classes = []
        for x_index in range(0, len(results)):
            if results[x_index] == [1, 0, 0]:
                a = 0
            elif results[x_index] == [0, 1, 0]:
                a = 1
            else:
                a = 2
            # print(str(results[x_index]) + " class: " + str(y[x_index]) + " index: " + str(x_index))
            classes.append(a)
        # print(classes)
        correct = 0
        for x in range(0, len(classes)):
            if classes[x] == y[x]:
                correct += 1
        print(str((correct/150.0) * 100) + "% accurate")

    # clf classifier from skLearn
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=3)

    y = y.ravel()
    train_y = np.array(y).astype(int)

    clf.fit(normalized_X, train_y)
    clf.predict(normalized_X)
    print(clf.score(normalized_X, y))

    print("\nPima-indians:\n")

    array = np.genfromtxt("/Users/jeremy/Documents/cs450/pima-indians-diabetes.csv", delimiter=",")
    X = array[:, :-1]
    Y = array[:, -1:]
    normalized_X = preprocessing.normalize(X)
    normalized_X = np.insert(normalized_X, normalized_X.shape[1], -1, axis=1)
    num_cols = normalized_X.shape[1]
    neuralNet = NeuralNet(num_cols, [4, 3], normalized_X)
    for i in range(0, 200):
        neuralNet.run(Y, i)
        results = neuralNet.calculate_targets()
        # print(results)
        classes = []
        for x in results:
            if x == [1, 0, 0]:
                a = 0
            else:
                a = 1
            classes.append(a)
        correct = 0
        for x in range(0, len(classes)):
            if classes[x] == Y[x][0]:
                correct += 1
        print(str(((correct + 0.0)/(len(classes) + 0.0)) * 100.0) + "% accurate")

    # clf classifier from skLearn
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=3)

    y = Y.ravel()
    train_y = np.array(y).astype(int)

    clf.fit(normalized_X, train_y)
    clf.predict(normalized_X)
    print(clf.score(normalized_X, Y))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt
import numpy
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter


def read_file(type="csv", path="C:\\Users\\Jeremy\\Desktop\\iris.csv"):
    my_data = genfromtxt(path, delimiter=',')
    return my_data[:, :4], my_data[:, 4:]


def print_iris():
    iris = datasets.load_iris()

    # Show the data (the attributes of each instance)
    print(iris.data)

    # Show the target values (in numeric format) of each instance
    print(iris.target)

    # Show the actual target names that correspond to each number
    print(iris.target_names)


def knn(x_train,y_train,x_test,y_test,k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(x_train, y_train)
    predictions = model.predict(x_test)

    correct_answers = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            correct_answers += 1
    print("The KNN k = {} algorithm from sk_learn is {}% accurate.".format(k,correct_answers / len(predictions) * 100))


def main():
    # print_iris()
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    # x,y = read_file()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    classifier = GaussianNB()
    model = classifier.fit(x_train, y_train)
    targets_predicted = model.predict(x_test)

    correct_answers = 0
    for i in range(len(targets_predicted)):
        if targets_predicted[i] == y_test[i]:
            correct_answers += 1
    print("The Gaussian Classifier is {}% accurate.".format(correct_answers/len(targets_predicted) * 100))

    # New "Algorithm"
    classifier = HardCodedClassifier()
    model = classifier.fit(x_train, y_train)
    targets_predicted = model.predict(x_test)

    correct_answers = 0
    for i in range(len(targets_predicted)):
        if targets_predicted[i] == y_test[i]:
            correct_answers += 1
    print("The new Algorithm is {}% accurate.".format(correct_answers/len(targets_predicted) * 100))

    # read_file()

    print()
    print()
    print("STARTING KNN... \n\n\n")
    knn(x_train,y_train,x_test,y_test,1)
    knn(x_train,y_train,x_test,y_test,2)
    knn(x_train,y_train,x_test,y_test,3)
    knn(x_train,y_train,x_test,y_test,10)
    knn(x_train,y_train,x_test,y_test,30)

    print("\nSTARTING OWN ALGORITHM FOR KNN... \n")
    k_values = [1, 2, 3, 10, 70]
    classifier = KnnClassifier()
    model = classifier.fit(x_train, y_train)
    for k in k_values:
        targets_predicted = model.predict(x_test, k)
        correct_answers = 0
        for i in range(len(targets_predicted)):
                    if targets_predicted[i] == y_test[i]:
                        correct_answers += 1
        print("The new KNN Algorithm with k = {} is {}% accurate.".format(k, correct_answers/len(targets_predicted) * 100))


class HardCodedClassifier:
    def fit(self, data_train, targets_train):
        return HardCodedModel()


class HardCodedModel:
    def predict(self, data_test):
        target = []
        i = 0
        while i < len(data_test):
            target.append(0)
            i += 1
        return target


class KnnClassifier:
    def fit(self, data_train, targets_train):
        return KnnModel(data_train, targets_train)


class KnnModel:
    def __init__(self, data_train, targets_train):
        self.data_train = data_train
        self.targets_train = targets_train

# For each row in test_data
    # Find distance from row to every point in the training_data
        # No need to square root it
    # Save as distance array
    # Find the k lowest distances
    # Classify as most frequent class as K

    def predict(self, data_test, k):
        target = []
        distances = []
        for row in data_test:
            train_row_number = 0
            for train_row in self.data_train:
                distance = 0
                for i in range(4):
                    distance += (row[i] - train_row[i]) ** 2
                distances.append([distance, self.targets_train[train_row_number]])
                train_row_number += 1
            distances = sorted(distances, key=lambda x: x[0])
            closest = []
            for i in range(k):
                closest.append(distances[i][1])
            target.append(max(set(closest), key=closest.count))
            distances.clear()
        return target


if __name__ == "__main__":
    main()

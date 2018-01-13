from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt


def read_file(type="csv", path="C:\\Users\\Jeremy\\Desktop\\iris.csv"):
    my_data = genfromtxt(path, delimiter=',')
    return my_data[:,:4], my_data[:, 4:]


def print_iris():
    iris = datasets.load_iris()

    # Show the data (the attributes of each instance)
    print(iris.data)

    # Show the target values (in numeric format) of each instance
    print(iris.target)

    # Show the actual target names that correspond to each number
    print(iris.target_names)


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

    read_file()


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


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


def read_file(headers, path):
    path = "C:\\Users\\Jeremy\\Desktop\\" + path
    # Define the headers since the data does not have any
    if path == "auto-mpg.csv":
        df = pd.read_csv(path,
                         header=None, names=headers, na_values="0", delim_whitespace=True)
    else:
        df = pd.read_csv(path,
                     header=None, names=headers, na_values="0")

    if path == "car.csv":
        cleanup_nums = {"doors": {"5more": 5}, "persons": {"more": 6}}
        df.replace(cleanup_nums, inplace=True)
        df["maint"] = df["maint"].astype('category').cat.codes
        df["buying"] = df["buying"].astype('category').cat.codes
        df["lug_boot"] = df["lug_boot"].astype('category').cat.codes
        df["safety"] = df["safety"].astype('category').cat.codes
        df["class"] = df["class"].astype('category').cat.codes
    df = df.values
    a = df[:, -1:]
    a = np.reshape(a,len(a))
    print(type(a[0]))
    return df[:, :-1], a


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

    def classification(closest):
        return max(set(closest), key=closest.count)

    def regression(closest):
        return np.mean(closest)

    headers_cars = ["buying", "maint", "doors", "persons", "lug_boot",
               "safety", "class"]
    headers_diabetes = ["timespregnant", "glucose", "bloodpressure", "skinfold", "insulin",
               "bmi", "pedigree", "age", "class"]
    headers_mpg = ["mpg", "cylinders", "displacement", "horsepower", "weight",
               "acceleration", "modelyear", "origin", "name"]

    # UCI: Car Evaluation data set

    x,y = read_file(headers_cars,"car.csv")
    split = KFold.split(x,y)
    main.x_train = None
    main.x_test = None
    main.y_train = None
    main.y_test = None
    for train_index, test_index in split:
        main.x_train, main.x_test = x[train_index], x[test_index]
        main.y_train, main.y_test = y[train_index], y[test_index]

    print()
    print()
    print("STARTING KNN... \n\n\n")
    knn(main.x_train,main.y_train,main.x_test,main.y_test,1)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,2)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,3)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,10)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,30)

    print("\nSTARTING OWN ALGORITHM FOR KNN... \n")
    k_values = [1, 2, 3, 10, 70]
    classifier = KnnClassifier()
    model = classifier.fit(main.x_train, main.y_train)
    for k in k_values:
        targets_predicted = model.predict(main.x_test, k, classification)
        correct_answers = 0
        for i in range(len(targets_predicted)):
                    if targets_predicted[i] == main.y_test[i]:
                        correct_answers += 1
        print("The new KNN Algorithm with k = {} is {}% accurate.".format(k, correct_answers/len(targets_predicted) * 100))

    # Pima Indian Diabetes data set

    x,y = read_file(headers_diabetes,"pima-indians-diabetes.csv")
    split = KFold.split(x,y)
    main.x_train = None
    main.x_test = None
    main.y_train = None
    main.y_test = None
    for train_index, test_index in split:
        main.x_train, main.x_test = x[train_index], x[test_index]
        main.y_train, main.y_test = y[train_index], y[test_index]

    print()
    print()
    print("STARTING KNN... \n\n\n")
    knn(main.x_train,main.y_train,main.x_test,main.y_test,1)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,2)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,3)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,10)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,30)

    print("\nSTARTING OWN ALGORITHM FOR KNN... \n")
    k_values = [1, 2, 3, 10, 70]
    classifier = KnnClassifier()
    model = classifier.fit(main.x_train, main.y_train)
    for k in k_values:
        targets_predicted = model.predict(main.x_test, k, classification)
        correct_answers = 0
        for i in range(len(targets_predicted)):
                    if targets_predicted[i] == main.y_test[i]:
                        correct_answers += 1
        print("The new KNN Algorithm with k = {} is {}% accurate.".format(k, correct_answers/len(targets_predicted) * 100))

    # Automobile MPG

    x,y = read_file(headers_mpg,"auto-mpg.csv")
    split = KFold.split(x,y)
    main.x_train = None
    main.x_test = None
    main.y_train = None
    main.y_test = None
    for train_index, test_index in split:
        main.x_train, main.x_test = x[train_index], x[test_index]
        main.y_train, main.y_test = y[train_index], y[test_index]

    print()
    print()
    print("STARTING KNN... \n\n\n")
    knn(main.x_train,main.y_train,main.x_test,main.y_test,1)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,2)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,3)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,10)
    knn(main.x_train,main.y_train,main.x_test,main.y_test,30)

    print("\nSTARTING OWN ALGORITHM FOR KNN... \n")
    k_values = [1, 2, 3, 10, 70]
    classifier = KnnClassifier()
    model = classifier.fit(main.x_train, main.y_train)
    for k in k_values:
        targets_predicted = model.predict(main.x_test, k, regression)
        correct_answers = 0
        for i in range(len(targets_predicted)):
                    if targets_predicted[i] == main.y_test[i]:
                        correct_answers += 1
        print("The new KNN Algorithm with k = {} is {}% accurate.".format(k, correct_answers/len(targets_predicted) * 100))


class KnnClassifier:
    def fit(self, data_train, targets_train):
        return KnnModel(data_train, targets_train)


class KnnModel:
    def __init__(self, data_train, targets_train):
        self.data_train = data_train
        self.targets_train = targets_train

    def predict(self, data_test, k, funct):
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
            target.append(funct(closest))
            distances.clear()
        return target


if __name__ == "__main__":
    main()

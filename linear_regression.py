import numpy as np
import pandas as pd
from math import floor

def load_data(path = "linearRegression_carPrice.csv"):
    data = pd.read_csv(path)
    dataset = data.values
    return dataset

def int_encoder(dataset):
    m = dataset.shape[0]
    output = np.zeros((m,0))

    for column in dataset.T:

        if not isinstance(column[1], str):
            column = np.reshape(column, (-1,1))
            output = np.append(output, column, axis=1)
            continue
        uniques = np.unique(column)

        for i in range(len(uniques)):
            column = np.where(column==uniques[i],i, column)

        column = np.reshape(column, (-1,1))
        output = np.append(output, column, axis=1)
#     print(output)
    return output


def oneHat_encoder(dataset):
    m = dataset.shape[0]
    output = np.zeros((m,0))

    for column in dataset.T:

        if not isinstance(column[1], str):
            column = np.reshape(column, (-1,1))
            output = np.append(output, column, axis=1)
            continue

        uniques = np.unique(column)

        for i in range(len(uniques)):

            temp = np.where(column==uniques[i],1, 0)
            temp = np.reshape(temp, (-1,1))
            output = np.append(output, temp, axis=1)
    return output

def partition_data(dataset,train_percent = 0.6, CV_percent = 0.2, test_percent = 0.2):
    m = dataset.shape[0]
    train_number = floor(train_percent * m)
    CV_number = floor(CV_percent * m)
    test_number = floor(test_percent * m)
    # print(type(dataset))
    train_data = dataset[0:train_number, :]
    CV_data = dataset[train_number:CV_number+train_number, :]
    test_data = dataset[CV_number+train_number:, :]

    return train_data, CV_data, test_data

def compute_cost(X, y, theta):
    m = X.shape[0]
    yhat = np.dot(X, theta)
    lost = np.power(yhat- y, 2)
    cost = 1/m * np.sum(lost)
    # print(cost)
    return cost


def one_variable_linear_regression(dataset,inital_theta, itteration = 1500, alpha = 0.01):
    assert dataset.shape[1] == 2

    X = np.append(np.zeros((m,1)) + 1, dataset[:, 0], axis = 1) # add x0
    y = dataset[:,1]
    # thata = np.zeros((2,1))


if __name__ == '__main__':
    # dataset = np.array([[1,"A", "Aa", 10], [2,"B", "Bb", 20], [3,"C", "Cc", 30]], dtype=object)
    # print(int_encoder(dataset))
    # print(oneHat_encoder(dataset))

    dataset = load_data()
    train_data, CV_data, test_data = partition_data(dataset)
    print (CV_data, CV_data.shape)

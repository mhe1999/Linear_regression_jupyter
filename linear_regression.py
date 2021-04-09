import numpy as np
import pandas as pd
from math import floor
import matplotlib.pyplot as plt

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
    lost = np.power(yhat - y, 2)
    cost = 1/(2*m) * np.sum(lost)
    # print(cost)
    return cost

def cal_gradients(X, y, theta):
    m = X.shape[0]
    yhat = np.dot(X, theta)
    # print("yhat:", yhat)
    gradients = 1/m * np.dot(X.T,(yhat - y))
    # print("gradiants:", gradients)
    # print("\n\n\ny:",y.shape)
    # print("yhat:", yhat.shape)
    # print("minese:" ,(yhat - y). shape)
    # print("gradient:",gradients.shape)
    return gradients

def update_parameters(gradients, theta, alpha):
    theta = theta - alpha * gradients
    return theta

def feature_normalization(X):
    X_norm = X;
    mu = np.zeros((1, X.shape[1]));
    sigma = np.zeros((1, X.shape[1]));
    for i in range(X.shape[1]):
        mu[0,i] = np.mean(X[:, i])
        sigma[0,i] = np.std(X[:, i])
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def one_variable_linear_regression(dataset,inital_theta, itteration = 1500, alpha = 0.01, FN=0):
    assert dataset.shape[1] == 2
    m = dataset.shape[0]
    # print("m:", m)
    theta = inital_theta
    X = np.reshape(dataset[:, 0],(-1,1))
    if FN == 1:
         X, X_mu, X_sigma = feature_normalization(X)
    X = np.append(np.zeros((m,1)) + 1, X, axis = 1) # add x0
    y = np.reshape(dataset[:,1], (-1,1))
    # X_norm, X_mu, X_sigma = feature_normalization(X)
    # X_norm, X_mu, X_sigma = feature_normalization(X)
    # print("X:",X)
    # print("y:",y)
    # print("theta:",theta.shape)
    # thata = np.zeros((2,1))
    cost_save = np.array([])
    for i in range(itteration):
        gradients = cal_gradients(X, y, theta)
        theta = update_parameters(gradients, theta, alpha)
        cost = compute_cost(X, y, theta)
        # print("gradiants:")
        # print(gradients)
        # print("\ntheta:")
        # print(theta)
        # print("\ncost:")
        print(cost)
        cost_save= np.append(cost_save, cost)
    return cost_save, theta

def linear_regression(dataset,inital_theta, itteration = 1500, alpha = 0.01, FN=0):
    m = dataset.shape[0]
    theta = inital_theta
    X = dataset[:,0:-1]
    # print(dataset)
    # print(X)
    # print(dataset.shape)
    # print(X.shape)
    X, X_mu, X_sigma = feature_normalization(X)
    X = np.append(np.zeros((m,1)) + 1, X, axis = 1) # add x0
    # print(X)
    y = np.reshape(dataset[:,-1], (-1,1))
    # print(y)
    cost_save = np.array([])
    for i in range(itteration):
        gradients = cal_gradients(X, y, theta)
        theta = update_parameters(gradients, theta, alpha)
        cost = compute_cost(X, y, theta)
        # print(cost)
        cost_save= np.append(cost_save, cost)
    return cost_save, theta, X_mu, X_sigma

def predict(X, theta, mu, sigma):
    m = X.shape[0]
    X_norm = (X - mu) / sigma
    X_norm = np.append(np.zeros((m,1)) + 1, X_norm, axis = 1)
    yhat = np.dot(X_norm, theta)
    return yhat

if __name__ == '__main__':

    data_all = load_data()
    dataset = data_all
    # dataset = np.append(data_all[:, 18:24], np.reshape(data_all[:, 25], (-1,1)), axis=1)
    dataset = oneHat_encoder(dataset)
    # print(dataset)
    train_data, CV_data, test_data = partition_data(dataset,train_percent = 0.999, CV_percent = 0, test_percent = 0.1)
    n = train_data.shape[1]
    theta = np.zeros((n,1))
    c , t, mu, sigma = linear_regression(train_data, theta, 5000, 0.175)

    plt.plot(c)
    plt.show()

    # print(train_data[:, 0:-1])
    yhat = predict(test_data[:, 0:-1], t, mu, sigma)
    # print(c)
    print("y:",test_data[:, -1])
    print("yhat:", yhat)

    # train_data, CV_data, test_data = partition_data(dataset)
    # single
    # train_data = dataset
    # dataset = np.append(np.reshape(data_all[:, 21], (-1,1)), np.reshape(data_all[:, 25], (-1,1)), axis=1)
    # print(sv_dataset)
    # print (CV_data, CV_data.shape)
    # theta = np.zeros((2,1))
    # theta[0][0] = 5613
    # theta[1][0] = 187
    # c, t= one_variable_linear_regression(dataset, theta, 100, 0.00001)
    # print(t)
    # plt.plot(train_data[:, 21], train_data[:, 25], 'o')
    # plt.plot(c)
    # x1 = 5
    # y1 = t[0] + x1 * t[1]
    # x2 = 400
    # y2 = t[0] + x2 * t[1]
    # print(y1)
    # print(y2)
    # x1, y1 = [5, 12], [1, 4]
    # x2, y2 = [1, 10], [3, 2]
    # plt.plot(x1,y1,x2,y2, marker='o')
    # plt.plot(data_all[:, 21],data_all[:, 25], 'yo', data_all[:, 21], t[0] + data_all[:, 21] * t[1], '--k')
    # plt.show()
    # plt.plot(c)
    # plt.show()

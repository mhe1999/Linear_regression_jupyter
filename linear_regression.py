import numpy as np
import pandas as pd

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


dataset = np.array([[1,"A", "Aa", 10], [2,"B", "Bb", 20], [3,"C", "Cc", 30]], dtype=object)
print(int_encoder(dataset))
print(oneHat_encoder(dataset))

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd 
import pickle

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    train, test = data_split(df, 0.2)
    x_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    x_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    y_train = train[['infectionProb']].to_numpy().reshape(2060,)
    y_test = test[['infectionProb']].to_numpy().reshape(515,)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    with open('model.pkl', 'wb') as file:
        pickle.dump(clf, file)
    

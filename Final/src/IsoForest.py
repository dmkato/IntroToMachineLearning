import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(420)

class IsoForest:
    def __init__(self, real=False):
        self.clf = IsolationForest(max_samples=1000, random_state=rng, contamination=0.05)

    def train(self, train_data):
        X = [x[:-1] for x in train_data if x[-1]==0]
        self.clf.fit(X)

    def test(self, test_data):
        X = [x[:-1] for x in test_data]
        y = [x[-1] for x in test_data]
        return zip(self.clf.decision_function(X),[0 if x==-1 else x for x in self.clf.predict(X)])

    def predict(self, X):
        return zip(self.clf.decision_function(X),[0 if x==-1 else x for x in self.clf.predict(X)])

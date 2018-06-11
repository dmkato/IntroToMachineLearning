from sklearn.neural_network import MLPClassifier

class NeuralNet:
    def __init__(self):
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-8,
                            hidden_layer_sizes=(200,200,10), random_state=1)


    def train(self, data):
        """
        Fit the data using self.clf
        """
        X = [x[:-1] for x in data]
        y = [x[-1] for x in data]
        self.clf.fit(X, y)


    def test(self, data):
        """
        Test the classifier with a test dataset
        data:
        """
        X = [x[:-1] for x in data]
        y = [x[-1] for x in data]
        p = [(p[int(y)], y) for p, y in zip(self.clf.predict_proba(X), self.clf.predict(X))]
        return p

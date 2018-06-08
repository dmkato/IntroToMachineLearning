from sklearn import svm

class SVM:
    def __init__(self):
        self.clf = svm.SVC(C=0.07, kernel='poly', degree=2)

    def train(self, data):
        X = [x[:-1] for x in data]
        y = [x[-1] for x in data]
        self.clf.fit(X, y)

    def test(self, data):
        X = [x[:-1] for x in data]
        y = [x[-1] for x in data]
        p = zip(self.clf.decision_function(X), self.clf.predict(X))
        return p

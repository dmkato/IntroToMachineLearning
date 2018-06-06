from src.Data import Data
from src.Knn import Knn

def write_predictions(preds):
    p_string = '\n'.join(["{},{}".format(s, p) for s, p in preds])
    with open('pred.csv', 'w') as f:
        f.write(p_string)

if __name__ == "__main__":
    train_data = Data("general").to_instances()
    test_data = Data("test").to_instances()
    knn = Knn()
    knn.train(train_data)
    p = knn.test(test_data)
    write_predictions(p)

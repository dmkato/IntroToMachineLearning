from src.Data import Data
from src.Knn import Knn

def write_predictions(preds, test_data):
    p_str = '\n'.join(["{},{}".format(s, p) for s, p in preds])
    g_str = '\n'.join([str(int(t[-1])) for t in test_data])
    with open('pred.csv', 'w') as f:
        f.write(p_str)
    with open('gold.csv', 'w') as f:
        f.write(g_str)


if __name__ == "__main__":
    general_data = Data("general").to_instances()
    print(len(general_data))
    eval_len = 100
    train_data = general_data[eval_len:]
    test_data = general_data[:eval_len]
    # test_data = Data("test").to_instances()
    knn = Knn()
    knn.train(train_data)
    p = knn.test(test_data)
    write_predictions(p, test_data)

from src.Data import Data
from src.Knn import Knn

def write_predictions(preds, test_data):
    """
    Write predictions and Ground Truth files to current dir
    """
    p_str = '\n'.join(["{},{}".format(s, p) for s, p in preds])
    g_str = '\n'.join([str(int(t[-1])) for t in test_data])
    with open('pred.csv', 'w') as f:
        f.write(p_str)
    with open('gold.csv', 'w') as f:
        f.write(g_str)


def percent_positive(data_set):
    """
    Returns the percentage of data in the dataset that has a positive label
    """
    pos_data = [d for d in data_set if d[-1] == 1]
    return len(pos_data) / len(data_set)


if __name__ == "__main__":
    general_data = Data("general").to_instances()
    print("Percent Positive:", percent_positive(general_data))
    print("Data len:", len(general_data))

    eval_len = len(general_data) // 10
    train_data = general_data[eval_len:]
    test_data = general_data[:eval_len]
    # test_data = Data("test").to_instances()
    knn = Knn()
    knn.train(train_data)

    p = knn.test(test_data)

    write_predictions(p, test_data)

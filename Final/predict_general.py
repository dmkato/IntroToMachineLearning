from src.Data import Data
from src.KNN import KNN
from src.SVM import SVM

def write_predictions(preds, test_data):
    """
    Write predictions and Ground Truth files to current dir
    """
    p_str = '\n'.join(["{},{}".format(s, int(p)) for s, p in preds])
    g_str = '\n'.join([str(int(t[-1])) for t in test_data])
    with open('eval/pred.csv', 'w') as f:
        f.write(p_str)
    with open('eval/gold.csv', 'w') as f:
        f.write(g_str)


def percent_positive(data_set):
    """
    Returns the percentage of data in the dataset that has a positive label
    """
    pos_data = [d for d in data_set if d[-1] == 1]
    return len(pos_data) / len(data_set)


def run_model(model, train_data, test_data):
    """
    Returns a list of predicitons off of a model class model
    """
    model = model()
    model.train(train_data)
    p = model.test(test_data)
    return p


if __name__ == "__main__":
    model = SVM

    general_data = Data("general", subsample_rate=0.015).to_instances()
    eval_len = len(general_data) // 10
    train_data = general_data[eval_len:]
    test_data = general_data[:eval_len]
    p = run_model(model, train_data, test_data)
    write_predictions(p, test_data)

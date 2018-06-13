import sys
from src.Data import Data
from src.Knn import Knn
from src.Svm import Svm
from src.NeuralNet import NeuralNet
from src.IsoForest import IsoForest

def parse_args(args):
    """
    Check if args are correct
    """
    if len(args) == 1 and args[0] == "general":
        return args[0], None
    elif (len(args) == 2 and args[0] == "individual"):
        return args[0], args[1]
    print("Usage: python3 predict <type> <individual #>")
    print("   type: general, individual")
    print("   individual #: 2, 7>")
    exit()


def write_test_results(preds, test_data, type):
    """
    Write predictions and Ground Truth files to current dir
    """
    p_str = '\n'.join(["{},{}".format(s, int(p)) for s, p in preds])
    with open('eval/pred.csv', 'w') as f:
        f.write(p_str)

    if type == "prediction":
        return

    g_str = '\n'.join([str(int(t[-1])) for t in test_data])
    with open('eval/gold.csv', 'w') as f:
        f.write(g_str)


def write_predictions(preds, test_data, type):
    """
    Write predictions and Ground Truth files to current dir
    """
    p_str = '\n'.join(["{},{}".format(s, int(p)) for s, p in preds])
    with open('eval/pred.csv', 'w') as f:
        f.write(p_str)


def percent_positive(data_set):
    """
    Returns the percentage of data in the dataset that has a positive label
    """
    pos_data = [d for d in data_set if d[-1] == 1]
    return len(pos_data) / len(data_set)


def test_model(model, train_data, test_data):
    """
    Returns a list of predictions off of a model class model
    """
    model = model()
    model.train(train_data)
    p = model.test(test_data)
    return p


def run_model(model, train_data, test_data):
    """
    Returns a list of predictions off of a model class model
    """
    model = model()
    model.train(train_data)
    p = model.predict(test_data)
    return p


def partition_data(data):
    """
    Returns the data split split into a training set and a testing set
    """
    eval_len = len(data) // 10
    return data[eval_len:], data[:eval_len]


if __name__ == "__main__":
    model = NeuralNet
    type, i_num = parse_args(sys.argv[1:])
    train_data = Data(type, i_num).to_instances()
    pred_data = Data(type, i_num, real=True).to_instances()
    # t = test_model(model, train_data, test_data)
    p = run_model(model, train_data, pred_data)
    write_predictions(p, pred_data, type)

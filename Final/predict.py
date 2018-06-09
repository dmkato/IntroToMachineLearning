import sys
from src.Data import Data
from src.KNN import KNN
from src.SVM import SVM

def parse_args(args):
    """
    Check if args are correct
    """
    if len(args) == 1 and args[0] == "general":
        return args[0], None
    elif (len(args) == 2 and args[0] == "individual"):
        return args[0], args[1]
    print("Usage: python3 predict <type: general, individual> <individual#: 2, 7>")
    exit()


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


def partition_data(data):
    """
    Returns the data split split into a training set and a testing set
    """
    eval_len = len(data) // 10
    return data[eval_len:], data[:eval_len]


if __name__ == "__main__":
    model = SVM
    type, i_num = parse_args(sys.argv[1:])
    data = Data(type, i_num).to_instances()
    train_data, test_data = partition_data(data)
    p = run_model(model, train_data, test_data)
    write_predictions(p, test_data)
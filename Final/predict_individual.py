from src.Data import Data
from src.Knn import Knn
import sys

def write_predictions(preds, test_data):
    """
    Write predictions and Ground Truth files to current dir
    """
    p_str = '\n'.join(["{},{}".format(s, p) for s, p in preds])
    g_str = '\n'.join([str(int(t[-1])) for t in test_data])
    with open('eval/pred.csv', 'w') as f:
        f.write(p_str)
    with open('eval/gold.csv', 'w') as f:
        f.write(g_str)


def parse_args():
    """
    Check if args are correct
    """
    if len(sys.argv) < 2:
        print("Usage: python3 predict_individual <individual #>")
        exit()
    return sys.argv[1]


if __name__ == "__main__":
    i_num = parse_args()
    individual_data = Data("individual",
                            subsample_rate=0.1,
                            individual_num=i_num).to_instances()
    eval_len = len(individual_data) // 10
    train_data = individual_data[eval_len:]
    test_data = individual_data[:eval_len]
    # test_data = Data("test").to_instances()
    KNN = KNN()
    KNN.train(train_data)
    p = KNN.test(test_data)
    write_predictions(p, test_data)

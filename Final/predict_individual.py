from src.Data import Data
import sys

def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python3 predict_individual <individual #>")
        exit()
    return sys.argv[1]

if __name__ == "__main__":
    individual_num = parse_args()
    train_data = Data("individual", individual_num).to_instances()
    for i in train_data[:6]:
        print(i)

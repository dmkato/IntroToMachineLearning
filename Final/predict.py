from src.Data import Data

if __name__ == "__main__":
    train_data = Data("data/general/Subject_1.csv")
    for d in train_data:
        print(d)

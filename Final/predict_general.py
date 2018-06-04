from src.Data import Data

if __name__ == "__main__":
    train_data = Data("general").to_instances()
    for i in train_data[:5]:
        print(i)

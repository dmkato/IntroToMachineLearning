from src.Data import Data
from src.Knn import Knn

if __name__ == "__main__":
    train_data = Data("general").to_instances()
    test_data = Data("test").to_instances()
    # for i in test_data:
    #     print(i)
    knn = Knn()
    knn.train(train_data)
    print("Knn Accuracy:", knn.test(test_data))

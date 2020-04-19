from sklearn.metrics import accuracy_score
from utils.FileUtil import FileUtil
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils.KeyStrokeClassifierKNN import KeyStrokeClassifierKNN as Kc


class KeyStrokeClassifierKNN:
    """Class used for carrying out classification of test key stroke data.
    It takes in 3 arguments.
    -   The first is the file path to the csv file containing the dataset.
    -   The second id the out of sample test feature to be classified which
        is a comma separated string of all the columns (features).
    -   The third is the knn model test ratio which specifies what percentage
        of the dataset should be used as test in generating the knn model.
        It is any decimal between 0.0 to 1.0.
    -   The fourth argument is the size of 'n' neighbours to be used"""

    def __init__(self, dataset_file_path, test_feature_string, knn_model_test_ratio, neighbour_size):
        self.dataset_file_path = dataset_file_path
        self.test_feature_string = test_feature_string
        self.knn_model_test_ratio = knn_model_test_ratio
        self.neighbour_size = neighbour_size

    def fetch_classification(self):
        keystroke_data = pd.read_csv(self.dataset_file_path)
        data = keystroke_data.iloc[:, 0:38]
        le = preprocessing.LabelEncoder()
        encoded_value = le.fit_transform(keystroke_data.iloc[:, 38:39])
        target = keystroke_data['CLASS']
        sample_text_row = pd.DataFrame.transpose(pd.DataFrame(self.test_feature_string.split(",")))

        # sample_text_row = sample_text_row[0]
        data_train, data_test, target_train, target_test = train_test_split(data, target,
                                                                            test_size=self.knn_model_test_ratio,
                                                                            random_state=10)
        knn_model = KNeighborsClassifier(n_neighbors=self.neighbour_size)
        knn_model.fit(data_train, target_train)
        pred = knn_model.predict(data_test)
        prediction = knn_model.predict(sample_text_row)
        print("KNeighbors accuracy score : ", accuracy_score(target_test, pred))
        print(str(prediction[0]))
        return str(prediction[0])


def main():
    try:
        f = FileUtil()

        keystroke_data = pd.read_csv("data/keystroke-data.csv")
        data = keystroke_data.iloc[:, 0:40]
        le = preprocessing.LabelEncoder()
        #encoded_value = le.fit_transform(keystroke_data.iloc[:, 40:41])
        target = keystroke_data['CLASS']
        sample_text_row = pd.DataFrame.transpose(pd.DataFrame("0.078,0.101,0.105,0.093,0.096,0.080,0.093,0.091,0.093,0.077,0.085,0.064,0.082,0.094,0.091,0.080,0.106,0.162,0.099,0.081,0.065,0.083,0.095,0.092,0.079,0.102,0.133,0.030,0.036,0.090,0.000,0.087,0.000,0.092,0.101,0.000,2.936,5.818".split(",")))
        #sample_text_row = sample_text_row[0]
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=10)
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(data_train, target_train)
        pred = neigh.predict(data_test)
        pred2 = neigh.predict(sample_text_row)
        print("KNeighbors accuracy score : ", accuracy_score(target_test, pred))
        print(str(pred[0]))
        f.readfile("demo.txt")

    except Exception as e:
        a = 4
    file_path = "data/keystroke-data.csv"
    test_string = "0.078,0.101,0.105,0.093,0.096,0.080,0.093,0.091,0.093,0.077,0.085,0.064,0.082,0.094,0.091,0.080,0.106,0.162,0.099,0.081,0.065,0.083,0.095,0.092,0.079,0.102,0.133,0.030,0.036,0.090,0.000,0.087,0.000,0.092,0.101,0.000,2.936,5.818"
    customClassifier = Kc(file_path, test_string, 0.3, 3)
    print(customClassifier.fetch_classification())
    print("Main Method")


main()

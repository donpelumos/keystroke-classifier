import sys
from sklearn.metrics import accuracy_score
from utils.FileUtil import FileUtil
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class KeyStrokeClassifierKNN:
    """Class used for carrying out classification of test key stroke data.
    It takes in 4 arguments.
    -   The first is the file path to the csv file containing the dataset.
    -   The second id the out of sample test feature to be classified which
        is a comma separated string of all the columns (features).
    -   The third is the knn model test ratio which specifies what percentage
        of the dataset should be used as test in generating the knn model.
        It is any decimal between 0.0 to 1.0.
    -   The fourth argument is the size of 'n' neighbours to be used"""
    __base_path = sys.path[0]

    def __init__(self, dataset_file_path, test_feature_string, knn_model_test_ratio, neighbour_size):
        self.dataset_file_path = dataset_file_path
        self.test_feature_string = test_feature_string
        self.knn_model_test_ratio = knn_model_test_ratio
        self.neighbour_size = neighbour_size

    def fetch_classification(self):
        keystroke_data = pd.read_csv(self.dataset_file_path)
        data = keystroke_data.iloc[:, 0:40]
        le = preprocessing.LabelEncoder()
        #encoded_value = le.fit_transform(keystroke_data.iloc[:, 40:41])
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
        #print("KNeighbors accuracy score : ", accuracy_score(target_test, pred))
        #print(str(prediction[0]))
        return str(prediction[0])

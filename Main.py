from sklearn.metrics import accuracy_score

from utils.FileUtil import FileUtil
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def main():
    try:
        f = FileUtil()

        keystroke_data = pd.read_csv("data/keystroke-data.csv")
        data = keystroke_data.iloc[:, 0:38]
        le = preprocessing.LabelEncoder()
        encoded_value = le.fit_transform(keystroke_data.iloc[:, 38:39])
        target = keystroke_data['CLASS']
        sample_text_row = pd.DataFrame.transpose(pd.DataFrame("0.078,0.101,0.105,0.093,0.096,0.080,0.093,0.091,0.093,0.077,0.085,0.064,0.082,0.094,0.091,0.080,0.106,0.162,0.099,0.081,0.065,0.083,0.095,0.092,0.079,0.102,0.133,0.030,0.036,0.090,0.000,0.087,0.000,0.092,0.101,0.000,2.936,5.818".split(",")))
        #sample_text_row = sample_text_row[0]
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.35, random_state=10)
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(data_train, target_train)
        pred = neigh.predict(data_test)
        pred2 = neigh.predict(sample_text_row)
        print("KNeighbors accuracy score : ", accuracy_score(target_test, pred))
        print(str(pred2[0]))
        f.readfile("demo.txt")
    except Exception as e:
        a = 4
    print("Main Method")


main()

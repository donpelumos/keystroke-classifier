from sklearn.metrics import accuracy_score
from utils.FileUtil import FileUtil
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils.KeyStrokeClassifierKNN import KeyStrokeClassifierKNN as Kc


def main():
    try:
        f = FileUtil()
        f.readfile("demo.txt")

    except Exception as e:
        a = 4
    file_path = "data/keystroke-data.csv"
    test_string = "0.103,0.109,0.137,0.105,0.099,0.121,0.106,0.104,0.101,0.100,0.097,0.073,0.083,0.097,0.102,0.104,0.146,0.116,0.098,0.095,0.097,0.112,0.106,0.124,0.101,0.176,0.025,0.033,0.072,0.059,0.081,0.086,0.048,0.149,0.058,0.011,4.126,7.589,1.644,1.168"
    customClassifier = Kc(file_path, test_string, 0.3, 5)
    print(customClassifier.fetch_classification())
    print("Main Method")


main()

'''
Created on 2018/02/26

@author: matsunagi
'''

import csv
import pandas as pd
from sklearn.svm import SVC


def convert_str_to_index(original_data):
    # convert Sex
    original_data["Sex"] = original_data["Sex"].map({"male": 0, "female": 1})
    # print(original_data)
    return original_data


def main():
    train_dataname = "../data/train.csv"
    test_dataname = "../data/test.csv"

    train_data = pd.read_csv(train_dataname)
    # TODO: convert Pclass to class features
    feature_train_data = train_data[["Age", "Sex", "Pclass"]]
    convert_train_data = convert_str_to_index(feature_train_data)
    convert_train_data["Age"].fillna(convert_train_data.Age.mean(), inplace=True)
    train_labels = train_data["Survived"]
    trainer = SVC(random_state=0).fit(convert_train_data, train_labels)

    test_data = pd.read_csv(test_dataname)
    feature_test_data = test_data[["Age", "Sex", "Pclass"]]
    convert_test_data = convert_str_to_index(feature_test_data)
    convert_test_data["Age"].fillna(convert_test_data.Age.mean(), inplace=True)
    with open("titanic_submit.csv", "w") as savefile:
        saver = csv.writer(savefile)
        saver.writerow(["PassengerId", "Survived"])
        saver.writerows(zip(test_data["PassengerId"],
                            trainer.predict(convert_test_data)))
    # print(trainer.predict(convert_test_data))


if __name__ == '__main__':
    main()

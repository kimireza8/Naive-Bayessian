import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

training_dataset = pd.read_csv('milk_training.csv')
train_data = np.array(training_dataset)[:, :-1]
train_label = np.array(training_dataset)[:, -1]
print("Train data:", train_data)
print("Train label:", train_label)

datatest = pd.read_csv('milk_testing.csv')
test_data = np.array(datatest)[:, :-1]
test_label = np.array(datatest)[:, -1]
print("Test data:", test_data)
print("test label:", test_label)

classifier = GaussianNB()
classifier.fit(train_data, train_label)

# ypred = classifier.predict(test_data)
# print("ytpred: ", ypred)

# akurasi = accuracy_score(test_label, ypred)
# print("akurasi sebelum normalisasi: ", akurasi)

sc = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = sc.fit_transform(train_data)
print("Normalisasi training : ", train_data_scaled)

test_data_scaled = sc.transform(test_data)
print("Normalisasi testdata:", test_data_scaled)

classifier.fit(train_data_scaled, train_label)

ypred = classifier.predict(test_data_scaled)
print("ytpred: ", ypred)

akurasi = accuracy_score(test_label, ypred)
print("akurasi sesudah normalisasi : ",akurasi)
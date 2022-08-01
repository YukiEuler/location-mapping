import numpy as np
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

koor_x = [1,1,1,2,2,3,3,4,5,6,7,8,3]
koor_y = [1,3,4,2,5,2,4,2,4,1,3,4,5]
lokasi = [0,0,2,0,2,0,2,1,1,1,1,1,2]

le = preprocessing.LabelEncoder()

encoded_x = le.fit_transform(koor_x)
encoded_y = le.fit_transform(koor_y)
encoded_loc = le.fit_transform(lokasi)

kumpulan = list(zip(encoded_x, encoded_y))

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(kumpulan, encoded_loc)

for a in range(0,4):
    for b in range(0,3):
        predicted= knn.predict([[a,b]])
        print(predicted)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# %matplotlib inline

df = pd.read_csv('/content/IRIS.csv')
df.head()

df.describe()

df.info()

sns.pairplot(df, hue='species')

data = df.values

X = data[:,0:4]
Y = data[:,4]

# split the data to train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

from sklearn.svm import SVC

model_svc = SVC()
model_svc.fit(X_train, Y_train)

prediction1 = model_svc.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction1))

# Logistic Regression

from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(X_train, Y_train)

prediction2 = model_LR.predict(X_test)
# calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction2)*100)
for i in range(len(prediction1)):
  print(Y_test[i], prediction1[i])

# Decision tree classifier

from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, Y_train)

prediction3 = model_svc.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction3))

from sklearn.metrics import classification_report
print(classification_report(Y_test, prediction2))

X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
prediction = model_svc.predict(X_new)
print("prediction of species : {}".format(prediction))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

titanic_trainData = pd.read_csv (r"C:\Users\Kingsama\Downloads\train.csv")
print (titanic_trainData)
titanic_trainData.head()
titanic_trainData.describe()
titanic_trainData.info()
titanic_trainData.isnull().sum()

test_data = pd.read_csv(r"C:\Users\Kingsama\Downloads\test.csv")
test_data.head()
test_data.describe()
test_data.info()
test_data.isnull().sum()

import matplotlib.pyplot as plt

women_survived = titanic_trainData.loc[titanic_trainData.Sex == 'female', 'Survived']
rate_women = sum(women_survived) / len(women_survived)
print("% of women who survived:", rate_women)

men_survived = titanic_trainData.loc[titanic_trainData.Sex == 'male', 'Survived']
rate_men = sum(men_survived) / len(men_survived)
print("% of men who survived:", rate_men)

features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(titanic_trainData[features])
y_train = titanic_trainData["Survived"]
X_test = pd.get_dummies(test_data[features])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# accuracy = round(model.score(X_train, y_train) * 100, 2)
# print("Accuracy is: %",accuracy)
print("Accuracy is: %",accuracy_score(y_train, model.predict(X_train)))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(r"C:\Users\Kingsama\Downloads\submission.csv", index=False)
print("Your submission was successfully saved!")





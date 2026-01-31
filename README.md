# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Read the Placement_data.csv file.And load the dataset.
3. Check the null and duplicate values.
4. Train and test the predicted value using logistic regression.
5. Calculate confusion matrix,accuracy,classification_matrix and predict.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Ritesh M Kamath
RegisterNumber: 25010798

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

data = data.drop(["sl_no", "salary"], axis=1)

data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})


X = data.drop("status", axis=1)
y = data["status"]


X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()

```

## Output:
<img width="757" height="762" alt="image" src="https://github.com/user-attachments/assets/f68963e4-0e29-460c-a905-6820a405af90" />
<img width="541" height="252" alt="image" src="https://github.com/user-attachments/assets/975e9776-eb94-4143-87a2-c4e763a9cf2b" />
<img width="718" height="562" alt="image" src="https://github.com/user-attachments/assets/6ab80e13-ec2e-4993-98e5-9e5a474df99e" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

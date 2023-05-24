# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sirisha Reddy
RegisterNumber:  212222230103
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
Data.head():



![image](https://github.com/22003264/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389139/fe7e2a52-b584-4686-8027-b1cb8ec0c136)

Data.info():




![image](https://github.com/22003264/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389139/e91f5083-b124-4bde-a531-a3a8f8f469cb)


isnull() and sum():





![image](https://github.com/22003264/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389139/13765c70-5817-4e4a-875b-191ff0b3a687)


Data Value Counts():



![image](https://github.com/22003264/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389139/01560d37-b412-446d-8648-705b39e7a7ea)



Data.head() for salary:


![image](https://github.com/22003264/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389139/459eba34-9924-4f09-a9fe-c3db2cb4b230)

x.head():


![image](https://github.com/22003264/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389139/9c7e8ea2-4992-464f-9c3e-8ade02113569)

Accuracy Value:


![image](https://github.com/22003264/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389139/7dd3fc29-5591-415e-b43c-bb3d9bcb392c)


Data Prediction:



![image](https://github.com/22003264/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389139/9a5343bc-3a31-4ab6-9e9d-96bad64d5909)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
data2 = dataset.drop(columns=['Loan_ID'],axis=1)

#Encooding catagorical in target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data2.Loan_Status = le.fit_transform(data2.Loan_Status)



#Imputing missing data
columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

for var in columns:
    data2[var].fillna(data2[var].mode()[0],inplace=True)

#Encoding catagorical Data
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for i in features:
    data2[i] = le.fit_transform(data2[i])

#Splittig into target variable and other columns
X = data2.iloc[:, :-1].values
Y = data2.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(cm)

#Test data

testdata = pd.read_csv('test.csv')

testdata = dataset.drop(columns=['Loan_ID',],axis=1)

#Imputing missing data
columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

for var in columns:
    testdata[var].fillna(testdata[var].mode()[0],inplace=True)

#Encoding catagorical Data
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for i in features:
    testdata[i] = le.fit_transform(testdata[i])
    
#Splittig into target variable and other columns
Xt = testdata.iloc[:, :-1].values
Yt = testdata.iloc[:, -1].values

Yt = le.fit_transform(Yt)

#regressor building
new_reg = LogisticRegression()
new_reg.fit(X,Y)

#Final predictions

results = new_reg.predict(Xt)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y,results)
print(cm)
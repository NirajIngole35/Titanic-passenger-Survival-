import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# step 1 info
ds = pd.read_csv(r"C:\Users\HP\Downloads\titanic.csv")
print(ds)
print(ds.head(10))
print(ds.tail(10))
print(ds.describe())
print(ds.info())
print(ds.isnull().sum())
print('###################################')
# step2 value count
print(ds['Sex'].value_counts())
print(ds['Survived'].value_counts())
print(ds["Pclass"].value_counts())
print(ds['Embarked'].value_counts())
print('###################################')
#step 3Converting genders into 0 or 1
gender = {"male": 0, "female": 1}
ds['Sex'] = ds['Sex'].map(gender)
#Convering Embarked feature into numeric data
ports = {"S": 0, "C": 1, "Q": 2}
ds['Embarked'] = ds['Embarked'].map(ports)
# step4 DISTRUBUTION ANALYSIS  graph
gender_data=ds['Sex'].value_counts()
survived_data=ds['Survived'].value_counts()
pclass_data=ds["Pclass"].value_counts()
embarked_data=ds['Embarked'].value_counts()
gender_label=['male','female']
survived_label=['Survived','died']
pclass_label=['1','2','3']
embarked_label=['s','c','q']
plt.figure(figsize=(10,5))

plt.suptitle('analysis')
plt.subplot(2,2,1)
plt.scatter('gender_data','gender_label',color='blue')
plt .grid(True)

plt.subplot(2,2,2)
plt.scatter('survived_data','survived_label',color='green')
plt .grid(True)

plt.subplot(2,2,3)
plt.scatter('embarked_data','embarked_label',color='black')
plt .grid(True)

plt.subplot(2,2,4)
plt.scatter('pclass_data','pclass_label',color='red')
plt .grid(True)
plt .show()

#step 5 machine l.
ds['Age'].fillna(ds['Age'].mean(),inplace=True)
ds['Sex'].fillna(ds['Sex'].mean(),inplace=True)
ds['SibSp'].fillna(ds['SibSp'].mean(),inplace=True)
ds['Embarked'].fillna(ds['Embarked'].mean(),inplace=True)
ds['Survived'].fillna(ds['Survived'].mean(),inplace=True)


#step 6 droup the column

ds = ds.drop(['Cabin'], axis=1)
ds = ds.drop(['Fare'], axis=1)
ds = ds.drop(['PassengerId'], axis=1)
ds = ds.drop(['Name'], axis=1)
ds = ds.drop(['Parch'], axis=1)
ds = ds.drop(['Ticket'], axis=1)
ds = ds.drop(['Pclass'], axis=1)
print(ds)

#step 7- i/p vareable(replace)
ds.replace({'Sex': {'male': 0, 'female' : 1}})
ds.replace({'Embarked': {'s': 0, 'c': 1, 'q': 2}})

#step 7- splite data in x,y
x=DataFrame(ds,columns=['Sex','Age','SibSp','Embarked'])
y=DataFrame(ds,columns=['Survived'])

#step 7- train ,test

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

#step  8 -model and predict
model=LogisticRegression()
model.fit(x_train, y_train)

pre=model.predict(x_test)
print(pre)

#step  9 out put
print(accuracy_score(y_test,pre))
print(classification_report(y_test,pre))
print(confusion_matrix(y_test,pre))
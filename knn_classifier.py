import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

data=pd.read_csv("diabetes.csv")
# data.info(verbose=True)
# print(data.describe())

data_copy=data.copy(deep=True)
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

data_copy.loc[:, 'Glucose'] = data_copy['Glucose'].fillna(data_copy['Glucose'].mean())
data_copy.loc[:, 'BloodPressure'] = data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean())
data_copy.loc[:, 'SkinThickness'] = data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median())
data_copy.loc[:, 'Insulin'] = data_copy['Insulin'].fillna(data_copy['Insulin'].median())
data_copy.loc[:, 'BMI'] = data_copy['BMI'].fillna(data_copy['BMI'].median())

# sns.pairplot(data_copy,hue='Outcome')
# plt.show()

# plt.figure(figsize=(12,10))
# p = sns.heatmap(data_copy.corr(), annot=True, cmap='RdYlGn')
# plt.show()

sc=StandardScaler()
x=sc.fit_transform(data_copy.drop(['Outcome'],axis=1))
y=data_copy['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=42,stratify=y)

# test_score=[]
# train_score=[]
#
# for i in range(1,15):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(x_train,y_train)
#     test_score.append(knn.score(x_test,y_test))
#     train_score.append(knn.score(x_train,y_train))
#
# p= sns.lineplot(test_score,marker='*')
# p=sns.lineplot(train_score,marker='o')
# plt.show()
#
#best value of n_neighbors=10

knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
score=knn.score(x_test,y_test)
print(score)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
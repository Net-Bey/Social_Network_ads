#importing numpy to perform maths
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


dataset = pd.read_csv("Social_Network_Ads.csv")

x= dataset.iloc[:,0:4].values
y=dataset.iloc[:,4:5].values

# Encoding the independent variable (X)

#importing preprocessing from SKLearn 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#importing the compose sub_library from sklearn 
from sklearn.compose import make_column_transformer 

#defining the functions of the make_column_transformer
column_changer =make_column_transformer((OneHotEncoder(),[1]),remainder='passthrough')

#assigning the variable x to the column_changer 
x = column_changer.fit_transform(x)


#splitting the dataset into train data and test data 
from sklearn.model_selection import train_test_split

#creating variable to store x_train,x_test and y_train,y_test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler

#assigning the StandardScaler to a variable sc_x
sc_x = StandardScaler()

#fitting and transforming the x train and the x test
x_train=sc_x.fit_transform(x_train)

#fitting  x_test
x_test=sc_x.transform(x_test)
 

#training the decision tree module
from sklearn.tree import DecisionTreeRegressor

# creating a variable and assigning the decision tree regression algorithm
Janet_social_module=DecisionTreeRegressor(random_state=1)

#training a module Janet with x_train and y_train 
Janet_social_module.fit(x_train,y_train)

#making a prediction
prediction_result=Janet_social_module.predict(x_test)

prediction_result 


#this is to evaluate the module
from sklearn.metrics import accuracy_score

#creating a variable to store the accuracy
score=accuracy_score(y_test,prediction_result)

score
print(score*100,'%')

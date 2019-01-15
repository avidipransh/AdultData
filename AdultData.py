"""
Name Of Author: Avi Dipransh
Date:13/01/2019
Description: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)) 

Prediction task is to determine whether a person makes over 50K a year. 


Attribute Information:

Listing of attributes: 

>50K, <=50K. 

age: continuous. 
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
fnlwgt: continuous. 
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
education-num: continuous. 
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
sex: Female, Male. 
capital-gain: continuous. 
capital-loss: continuous. 
hours-per-week: continuous. 
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


"""
#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from io import StringIO
from urllib.request import urlopen
import six

#Scrapping data from the website.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
html = urlopen(url)
soup = BeautifulSoup(html , 'lxml')

text = soup.get_text()
data = StringIO(text)
 
#creating the dataset
dataset = pd.read_csv(data , sep = ",")

#splitting the independent and dependent variales.
X = dataset.iloc[: , :-1].values
x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 14].values



#encoding categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#fnction to label encode data
def labelEncode(X):
	num_vars = 14		#since number of independent attributes are 14
	i = 0 
	for i in range(0 , num_vars):
		if(isinstance(x[0 , i] , six.string_types)):			#checking if the data is categorical or not.
			encoder = LabelEncoder()
			X[: , i] = encoder.fit_transform(X[: , i])
	return X

#function to one hot encode categorical data
def oneHotEncoder(X):
	num_vars = 14 	#since number of independent attributes originally are 14
	i = 0
	for i in range(num_vars):
		if((isinstance(x[0 , i] , six.string_types)) and (i != 9) ):		#Checking weather the given entry is a string or not. And not checking for coloumn 9 as it is a binary entry, MALE/FEMALE
			ohe = OneHotEncoder(categorical_features = [i])		#if it is, the encoding the entire coloumn using ohe
			X = ohe.fit_transform(X).toarray()
	return X

X = labelEncode(X)

#taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, : - 1])
X[:, : - 1] = imputer.transform(X[:, : - 1])

X_dup = X
df = pd.DataFrame(X_dup)



X = oneHotEncoder(X)
X_dup = X
df = pd.DataFrame(X_dup)

ohe = OneHotEncoder(categorical_features = [14])
X = ohe.fit_transform(X).toarray()
df = pd.DataFrame(X)

#splitting the data into training and test sets
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3)

#using the NB Classifier to predict data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train , y_train)
y_pred = classifier.predict(X_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
 
#finding the accuracy of the model.
sum1 = cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1]
cor = cm[0][0] + cm[1][1]
accuracy = (cor/sum1)*100 




















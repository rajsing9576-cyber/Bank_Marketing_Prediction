# Importing all the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



# load the dataset from csv file 
data = pd.read_csv('bank-full.csv', sep=";")

# print first 5 rows
print(data.head())

# Data Visulization
# Histogram
print(data['age'].hist())
plt.show()


# Box-plot
print(plt.boxplot(data['balance']))
plt.show()

# scatter plot
print(data['job'].value_counts().plot(kind='bar'))
plt.show()


# Correlation heatmap
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.show()


# Pie-chart
data['y'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()

# rename the categorical_data into numerical_dataset
le = LabelEncoder()

categorical_columns=[
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "poutcome",
    "y",
    "month"

]

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])


# print first 5 rows
data.head()



# Droping the data into X and Y
X =data.drop("y", axis=1)
Y = data["y"]



# printing the X
print(X)


# printing the X
print(Y)

# Splitting the data into Training data and Testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

# shape of X
X.shape

# shape of X_train
X_train.shape

# shape of X_test
X_test.shape

# Support Vector Machine Model
SVC = SVC()


# Train the Model
SVC.fit(X_train,Y_train)

# Train the model in train dataset
train_data = SVC.predict(X_train)

# Accuracy on training data in SVC
training_data_accuracy = accuracy_score(Y_train, train_data)

print("Accuracy on training_data in SVC model:", training_data_accuracy)


# Test the model in test dataset
test_data = SVC.predict(X_test)

# Accuracy on testing data in SVC
testing_data_accuracy = accuracy_score(Y_test, test_data)

print("Accuracy on testing_data in SVC model:", testing_data_accuracy)

# LogisticRegression Model
Le = LogisticRegression()

# Train the model 
Le.fit(X_train,Y_train)


# Train the Model in training dataset
train_data = Le.predict(X_train)

# Accuracy on training dataset in LogisticRegression
training_data_accuracy = accuracy_score(Y_train,train_data)

print("Accuracy on training_data in LogisticRegression model:",training_data_accuracy)

# Test the Model in testing dataset
test_data = Le.predict(X_test)

# Accuracy on testing_dataset in LogisticRegression
testing_data_accuracy = accuracy_score(Y_test,test_data)

print("Accuracy on testing_data in LogistcRegression model:",testing_data_accuracy)

# RandomForestCLassifier Model
Rd = RandomForestClassifier()

# Train the Model 
Rd.fit(X_train, Y_train)

# Train a Model in training dataset
train_data = Rd.predict(X_train)


# Accuracy on training data in RandomForestClassifier Model
training_data_accuracy = accuracy_score(Y_train, train_data)


print("Accuracy on training_data in RandomForestClassifier model:",training_data_accuracy)

# Train a Model in testing dataset
test_data = Rd.predict(X_test)

# Accuracy on testing data in RandomForestClassifier Model
testing_data_accuracy = accuracy_score(Y_test, test_data)


print("Accuracy on testing_data in RandomForestClassifier model:",testing_data_accuracy)


# DecisionTreeClassifier Model
Dd = DecisionTreeClassifier()


# Train a Model
Dd.fit(X_train, Y_train)


# Train a Model on training dataset
train_data = Dd.predict(X_train)


# Accuracy on training_data in DecisionTreeClassifier Model
training_data_accuracy = accuracy_score(Y_train, train_data)

print("Accuracy on training_data in DecisionTreeClassifier model:", training_data_accuracy)


# Train a Model on testing dataset
test_data = Dd.predict(X_test)


# Accuracy on testing_data in DecisionTreeClassifier Model
testing_data_accuracy = accuracy_score(Y_test, test_data)

print("Accuracy on testing_data in DesicionTreeClassifier model:",testing_data_accuracy)


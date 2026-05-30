import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# load the Dataset
data = pd.read_csv('Bank.csv')
# print first 5 rows of dataset
print(data.head())

# check the info of dataset
print(data.info())

# checking the missing values in dataset
print(data.isnull().sum())

# Data Visulaization
print(data['education'].hist())
print(data['balance'].hist())

# Count the value of job columns 
print(data['job'].value_counts())


# Encoding the Alphabatecial data to Numerical data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorial_columns = [
    "job"
    "marital"
    "education"
    "default"
    "housing"
    "loan"
    "contact"
    "month"
    "poutcome"
    "y"
]

for col in categorial_columns:
    data[col] = le.fit_transform(data[col])

    
# Check the shape of dataset
print(data.shape)

# Ploting the Scatter between age and balance
print(plt.scatter(data['age'], data['balance']))
print(plt.xlabel('Age'))
print(plt.ylabel('Balance'))
print(plt.title('Age vs Balance'))
print(plt.show())




# Plotting the Scatter between age and duration
print(plt.scatter(data['age'], data['duartion'], c='red'))
print(plt.xlabel('Age'))
print(plt.ylabel('Duration'))
print(plt.show())

# Drop the duration columns in the dataset
X = data.drop('education', axis=1)
Y = data['education']

print(X)
print(Y)




# Splitting the dataset into Training data and Testing data
X_train, Y_train, X_test, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(X.shape)

# @ Train the data into a SVC model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X = data.drop("y", axis=1)
Y = data["y"]


print(X.shape)
print(Y.shape)

X_train, Y_train, X_test, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

SVC = SVC()

SVC.fit(X_train, Y_train)

# Reshape the dataset
import pandas as pd

# reset indexes
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

# combine both
data = pd.concat([X, Y], axis=1)

# remove NAN rows
data = data.dropna()



# recreate X and Y from SAME dataframe
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]



# Check shapes
print(X.shape)
print(Y.shape)



# recreate the splitting the dataset into Training data and Testing data
from sklearn.model_selection import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


print(X_train.shape)
print(Y_train.shape)

# Train the model in SVC
train_data = SVC.predict(X_train)

# accuracy score of train_data in SVC
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Accuracy on training data in SVC model:", training_data_accuracy)


# Test the model in SVC
test_data = SVC.predict(X_test)

# acuracy score of test_data in SVC
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Accuracy on testing data in SVC model", testing_data_accuracy)


# Logistic Regression model
LogisticRegression = LogisticRegression()
LogisticRegression.fit(X_train, Y_train)

# Train the model
train_data = LogisticRegression.predict(X_train)

# accuarcy score of train_data in LogosticRegression
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Accuracy on training data in Logistic Regression model:", training_data_accuracy)



# Test the model
test_data = LogisticRegression.predict(X_test)

# accuracy score of test_data in LogisticRegression
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Acuuracy score on testing data in Logistic Regression model:", testing_data_accuracy)



# RandomForestClassifier model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Train the model in RandomForestCLassifier
train_data = RandomForestClassifier.predict(X_train)

# accuarcy score of train_data in RandomForestCLassifier
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Accuarcy score on training data in RandomForestClassifier model:", training_data_accuracy)


# Test the model in RandomForestCLassifier
test_data = RandomForestClassifier.predict(X_test)

# accuracy score on test_data in RandomForestCLassifier
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Accuracy score on testing data in RandomForestCLassifier model:", testing_data_accuracy)


# DecisionTreeClassifier model
DecisionTreeClassifier = DecisionTreeClassifier
DecisionTreeClassifier.fit(X_train, Y_train)

# Train the model in DecisionTreeCLassifier
train_data = DecisionTreeClassifier.predict(X_train)

# accuracy score on training data in DEcisionTreeClassifier
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Accuracy score on training data in DecisionTreeClassifier model:", training_data_accuracy)

# Test the model in DecisionTreeCLassifier
test_data = DecisionTreeClassifier.predict(X_test)

# accuarcy score on test_data in DecisionTreeClassofier
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Accuracy score on testing data in DecisionTreeCLassifier model:", testing_data_accuracy)





# Hence We train and test the data in 4 model
# .SVC
# .LogisticRegression
# .RandomForestCLassifer
# .DecisionTreeClassifier




# Coding Exercise 3: Encoding Categorical Data for Machine Learning
# 1: Import required libraries - Pandas, Numpy, and required classes for this task - ColumnTransformer, OneHotEncoder, LabelEncoder.
# 2: Start by loading the Titanic dataset into a pandas data frame. This can be done using the pd.read_csv function. The dataset's name is 'titanic.csv'.
# 3: Identify the categorical features in your dataset that need to be encoded. You can store these feature names in a list for easy access later.
# 4: To apply OneHotEncoding to these categorical features, create an instance of the ColumnTransformer class. Make sure to pass the OneHotEncoder() as an argument along with the list of categorical features.
# 5: Use the fit_transform method on the instance of ColumnTransformer to apply the OneHotEncoding.
# 6: The output of the fit_transform method should be converted into a NumPy array for further use.
# 7: The 'Survived' column in your dataset is the dependent variable. This is a binary categorical variable that should be encoded using LabelEncoder.
# 8.  Print the updated matrix of features and the dependent variable vector


# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = pd.read_csv('titanic.csv')
X = dataset.iloc[:,:]
y = dataset.iloc[:,1]

# Identify the categorical data

categorical_features = ['Sex', 'Embarked', 'Pclass']

# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), categorical_features)], remainder='passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer
X = np.array(ct.fit_transform(X))

# Convert the output into a NumPy array


# Use LabelEncoder to encode binary categorical data

le = LabelEncoder()
y = le.fit_transform(y)

# Print the updated matrix of features and the dependent variable vector
print (X)
print(y)


# Coding exercise 5: Feature scaling for Machine Learning
# 1. Import the necessary libraries for data preprocessing, including the StandardScaler and train_test_split classes.
# 2. Load the "Wine Quality Red" dataset into a pandas DataFrame. You can use the pd.read_csv function for this. Make sure you set the correct delimeter for the file.
# 3. Split your dataset into an 80-20 training-test set. Set random_state to 42 to ensure reproducible results.
# 4. Create an instance of the StandardScaler class.
# 5. Fit the StandardScaler on features from the training set, excluding the target variable 'Quality'.
# 6. Use the "fit_transform" method of the StandardScaler object on the training dataset.
# 7. Apply the "transform" method of the StandardScaler object on the test dataset.
# 8. Print your scaled training and test datasets to verify the feature scaling process.



# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset

# Load the Wine Quality Red dataset
dataset=pd.read_csv('winequality-red.csv', delimiter=";")

# Separate features and target
X = dataset.drop(columns=['quality']) 
y = dataset['quality']

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Create an instance of the StandardScaler class
sc= StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
X_train = sc.fit_transform(X_train)

# Apply the transform to the test set
X_test = sc.transform(X_test)

# Print the scaled training and test datasets
print(X_train)
print(X_test)

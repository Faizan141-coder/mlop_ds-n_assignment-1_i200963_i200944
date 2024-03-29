# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv('audi.csv')
X = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values
Y = df.iloc[:, [2]].values.ravel()  # Use ravel() to convert Y into a 1D array

# Label Encoding
le1 = LabelEncoder()
X[:, 0] = le1.fit_transform(X[:, 0])
le2 = LabelEncoder()
X[:, -4] = le2.fit_transform(X[:, -4])

# One Hot Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = ct.fit_transform(X)

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting Dataset into Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training Model
regression = RandomForestRegressor(random_state=0)
regression.fit(X_train, Y_train)
y_pred = regression.predict(X_test)

# Testing result
print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Calculating Accuracy
print("R2 Score:", r2_score(Y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(Y_test, y_pred))

# Reshape to 2D
y_pred = np.reshape(y_pred, (-1, 1))

# Making Pandas DataFrame
mydata = np.concatenate((Y_test.reshape(-1, 1), y_pred), axis=1)
dataframe = pd.DataFrame(mydata, columns=['Real Price', 'Predicted Price'])
print(dataframe)


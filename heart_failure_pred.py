import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Importing Dataset
dataset = pd.read_csv('heart.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Data Preprocessing
# Label Encoding
le1 = LabelEncoder()
le2 = LabelEncoder()
le6 = LabelEncoder()
le8 = LabelEncoder()
le10 = LabelEncoder()
x[:, 1] = le1.fit_transform(x[:, 1])
x[:, 2] = le2.fit_transform(x[:, 2])
x[:, 6] = le6.fit_transform(x[:, 6])
x[:, 8] = le8.fit_transform(x[:, 8])
x[:, 10] = le10.fit_transform(x[:, 10])

# Splitting Dataset into Training set and Test set
# flake8: noqa
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Dataset
# Training with Random Forest Classifier
model_randomforest = RandomForestClassifier()
model_randomforest.fit(X_train, Y_train)

model_logistic = LogisticRegression()
model_logistic.fit(X_train, Y_train)

model_kneighbors = KNeighborsClassifier()
model_kneighbors.fit(X_train, Y_train)

model_decision = DecisionTreeClassifier()
model_decision.fit(X_train, Y_train)

model_xgb = XGBClassifier()
model_xgb.fit(X_train, Y_train)

model_svm = SVC()
model_svm.fit(X_train, Y_train)

# Making Predictions
y_pred_logistic = model_logistic.predict(X_test)
y_pred_neighbors = model_kneighbors.predict(X_test)
y_pred_svm = model_svm.predict(X_test)
y_pred_decision = model_decision.predict(X_test)
y_pred_random = model_randomforest.predict(X_test)
y_pred_xgb = model_xgb.predict(X_test)

# Calculating accuracies
RandomForest_Accuracy = accuracy_score(Y_test, y_pred_random)
LogisticRegression_Accuracy = accuracy_score(Y_test, y_pred_logistic)
KNeighbors_Accuracy = accuracy_score(Y_test, y_pred_neighbors)
SVM_Accuracy = accuracy_score(Y_test, y_pred_svm)
Decision_Accuracy = accuracy_score(Y_test, y_pred_decision)
XGBoost_Accuracy = accuracy_score(Y_test, y_pred_xgb)

print("Random Forest Accuracy:", RandomForest_Accuracy)
print("Logistic Regression Accuracy:", LogisticRegression_Accuracy)
print("KNeighbors Accuracy:", KNeighbors_Accuracy)
print("SVM Accuracy:", SVM_Accuracy)
print("Decision Tree Accuracy:", Decision_Accuracy)
print("XGBoost Accuracy:", XGBoost_Accuracy)

# Save the trained model
joblib.dump(model_randomforest, 'heart_failure_pred_model.pkl')

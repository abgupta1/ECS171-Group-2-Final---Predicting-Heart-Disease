import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# # Load the CSV file into a DataFrame
# file_path = 'heart.csv'  # Replace with your file path
# data = pd.read_csv(file_path)
#
# # Encode categorical features
# le = LabelEncoder()
# for column in data.columns:
#     if data[column].dtype == 'object':
#         data[column] = le.fit_transform(data[column])
#
# # Split the data into features and target variable
# X = data.drop('HeartDisease', axis=1)
# y = data['HeartDisease']
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Initialize models
# models = {
#     "Random Forest Classifier": RandomForestClassifier(),
# }
#
# # Dictionary to hold evaluation metrics
# evaluation_metrics = {}
#
# # Train and evaluate each model
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     evaluation_metrics[name] = {"Accuracy": accuracy, "Precision": precision, "F1-Score": f1, "Recall": recall}
#
# # Output the evaluation metrics
# # for model_name, metrics in evaluation_metrics.items():
# #     print(f"{model_name}:")
# #     for metric_name, metric_value in metrics.items():
# #         print(f"  {metric_name}: {metric_value:.4f}")  # Formatting to 4 decimal places
# #     print()
#
#
# import pickle
# pickle.dump(model, open('model.pkl', 'wb'))
#
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[40, 130, 275, 0, 150, 0.0, 1, 1, 0, 0, 1, 0, 0, 0,1]])) #8:22
# packages
import sys
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load the CSV file into a DataFrame
file_path = 'heart.csv'  # Replace with your file path
data = pd.read_csv(file_path)

df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"], drop_first=True)
sc = StandardScaler()

# prints VIF results for predictor variables. Looking for VIF results less than 6
x = df_encoded.drop("HeartDisease", axis=1)
#x

# this model uses all of the columns in the dataset
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']
# splits data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(x,y , test_size=0.8, shuffle=True)
X_train = sc.fit_transform(X_train)
print(X_test)
print(y_test)
X_test = sc.transform(X_test)

# runs SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)
model_pred = model.score(X_test, y_test)
print(f'Prediction accuracy of model: {model_pred}')

# try1 = sc.transform([[40, 130, 275, 0, 150, 0.0, 1, 1, 0, 0, 1, 0, 0, 0,1]])
# print("trans",try1)
# print(model.predict(sc.transform([[40, 130, 275, 0, 150, 0.0, 1, 1, 0, 0, 1, 0, 0, 0,1]]))) #8:22
# print(model.predict(sc.transform([[54, 108, 267, 0, 167, 0.0, 0, 0, 1, 1, 0, 0, 0, 0,1]]))) #8:22
# print(model.predict(sc.transform([[60, 140, 281, 0, 118, 1.5, 1, 0, 0, 0, 0, 1, 1, 1,0]]))) #8:22
# print(model.predict(sc.transform([[48, 138, 214, 0, 108, 1.5, 1, 0, 0, 0, 1, 0, 1, 1,0]]))) #8:22
# print(model.predict(sc.transform([[70, 140, 214, 0, 108, 1.5, 1, 0, 0, 0, 1, 0, 1, 1,0]]))) #8:22

kernels = ['linear']
hyperparameters = [0.1]

for j in kernels:
    for i in hyperparameters:
        model = SVC(kernel=j, C=i)
        model.fit(X_train, y_train)
        model_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, model_pred)
        print(f'Kernel: {j}, C: {i}, Accuracy: {accuracy:.2f}')

import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


print(model.predict(scaler.transform([[40, 130, 275, 0, 150, 0.0, 1, 1, 0, 0, 1, 0, 0, 0,1]]))) #8:22
print(model.predict(scaler.transform([[54, 108, 267, 0, 167, 0.0, 0, 0, 1, 1, 0, 0, 0, 0,1]]))) #8:22
print(model.predict(scaler.transform([[60, 140, 281, 0, 118, 1.5, 1, 0, 0, 0, 0, 1, 1, 1,0]]))) #8:22
print(model.predict(scaler.transform([[48, 138, 214, 0, 108, 1.5, 1, 0, 0, 0, 1, 0, 1, 1,0]]))) #8:22
print(model.predict(scaler.transform([[70, 140, 214, 0, 108, 1.5, 1, 0, 0, 0, 1, 0, 1, 1,0]]))) #8:22


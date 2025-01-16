import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("heart.csv")

# Basic dataset inspection
print("Dataset Shape:", df.shape)
print("Missing Values:")
print(df.isnull().sum())

# Splitting features and target
target = 'target'  # Adjust based on dataset
y = df[target]
X = df.drop(columns=[target])

X

X.dtypes

y.value_counts()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)

rfe = RFE(estimator=model, n_features_to_select=3)  # You can change n_features_to_select based on your need
X_train_rfe = rfe.fit_transform(X_train, y_train)

model.fit(X_train_rfe, y_train)

# Evaluate the model on the test data using the selected features
X_test_rfe = rfe.transform(X_test)  # Transform test data to the same features as training data
y_pred = model.predict(X_test_rfe)

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



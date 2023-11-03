import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("D:/mine/Programs/5thsem/ml/Iris.csv")

num = df.select_dtypes(include=['float64']).columns 

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

df[num] = preprocessor.fit_transform(df[num])

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

for i in range(len(y_test)):
    print("Test ",i + 1,": Predicted ",y_pred[i],", Actual ",y_test.iloc[i])

correct_predictions = np.sum(y_pred == y_test)
wrong_predictions = np.sum(y_pred != y_test)

print("\nCorrect predictions:", correct_predictions)
print("Wrong predictions:", wrong_predictions)


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['SepalLengthCm'], y=X_test['SepalWidthCm'], hue=y_pred, palette='viridis')
plt.title("KNN Classifier Predictions")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

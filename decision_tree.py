import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Load the dataset and learn about it
df = pd.read_csv("/Users/burakg/Desktop/Boston_House_Price_Reggression_Problem/boston.csv")
df.head()
df.info()
columns = df.columns
print(columns)
print(df.shape)
#Train and test split
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_reg = DecisionTreeRegressor(random_state=42)
# Fit the model
tree_reg.fit(X_train, y_train)
y_pred = tree_reg.predict(X_test)
# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Calculate the root mean squared error
print(f"Root Mean Squared Error: {np.sqrt(mse)}")
# Feature importances
importances = tree_reg.feature_importances_
sorted_importances = np.sort(importances)[::-1]
sorted_indices = np.argsort(importances)[::-1]
# Print feature importances
for rank, i in enumerate(sorted_indices, start = 1):
    print(f"Rank {rank}:  Feature: {X.columns[i]}, importance : {importances[i]}")
    
#En yüksek importance değeri CRIM'den geliyor. Bu, özelliğin ev fiyatlarını en çok etkileyen faktör olduğunu gösteriyor.

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Gerçek Değerler (y_test)")
plt.ylabel("Tahmin Değerleri (y_pred)")
plt.title("Gerçek vs Tahmin Edilen Değerler")
plt.grid(True)
plt.show()
# R^2 Score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")
# R^2 skoru, modelin ne kadar iyi performans gösterdiğini ölçer. 1'e yakın bir değer, modelin veriyi iyi açıkladığını gösterir.
#R^2 Score: 0.8480530710805643
    


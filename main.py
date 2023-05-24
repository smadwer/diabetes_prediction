import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics

# load your data
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Encode categorical data
df = pd.get_dummies(df, drop_first=True)

# Remove rows with NaN values
df = df.dropna()

# Split data into predictors and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# Print out the mean squared error (mse)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

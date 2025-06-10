# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Read training and testing data
train_data = pd.read_csv('/workspace/ML2025Spring-hw2-public/train.csv')
test_data = pd.read_csv('/workspace/ML2025Spring-hw2-public/test.csv')

# Perform exploratory data analysis
print(train_data.head())
print(train_data.info())
print(train_data.describe())

# Handle missing values
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Scale the data
scaler = StandardScaler()
train_data[['cli_day1', 'ili_day1', 'cli_day2', 'ili_day2', 'cli_day3', 'ili_day3']] = scaler.fit_transform(train_data[['cli_day1', 'ili_day1', 'cli_day2', 'ili_day2', 'cli_day3', 'ili_day3']])
test_data[['cli_day1', 'ili_day1', 'cli_day2', 'ili_day2', 'cli_day3', 'ili_day3']] = scaler.transform(test_data[['cli_day1', 'ili_day1', 'cli_day2', 'ili_day2', 'cli_day3', 'ili_day3']])

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(train_data.drop('tested_positive_day3', axis=1), train_data['tested_positive_day3'], test_size=0.2, random_state=42)

# Define the model and hyperparameter space
model = RandomForestRegressor()
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}

# Perform hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_X, train_y)

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(train_X, train_y)

# Make predictions on the validation set
val_pred = best_model.predict(val_X)
print('Validation MSE:', mean_squared_error(val_y, val_pred))

# Make predictions on the testing set
test_pred = best_model.predict(test_data.drop('tested_positive_day3', axis=1))

# Save the predictions to a submission file
submission = pd.DataFrame({'id': test_data['id'], 'tested_positive_day3': test_pred})
submission.to_csv('/workspace/submission.csv', index=False)
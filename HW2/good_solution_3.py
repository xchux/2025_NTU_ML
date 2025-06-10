import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_csv('/workspace/ML2025Spring-hw2-public/train.csv')
test_df = pd.read_csv('/workspace/ML2025Spring-hw2-public/test.csv')

# Check for missing values
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Handle missing values
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Scale data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df.drop('tested_positive_day3', axis=1))
test_scaled = scaler.transform(test_df.drop('id', axis=1))

# Encode categorical variables
categorical_cols = train_df.select_dtypes(include=['object']).columns
numerical_cols = train_df.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

train_scaled = preprocessor.fit_transform(train_scaled)
test_scaled = preprocessor.transform(test_scaled)

# Split data into training and validation sets
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_df['tested_positive_day3'], test_size=0.2, random_state=42)

# Define models and hyperparameter grids
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor()
}

grids = {
    'Linear Regression': {
        'C': np.logspace(-5, 5, 11)
    },
    'Random Forest': {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 5, 10, 20]
    }
}

# Perform cross-validation and hyperparameter tuning
best_model = None
best_score = 0
for model_name, model in models.items():
    grid = grids[model_name]
    grid_search = GridSearchCV(model, grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_scaled, train_target)
    score = grid_search.best_score_
    if score > best_score:
        best_model = grid_search.best_estimator_
        best_score = score
    print(f'Model: {model_name}, Best Score: {score}')

# Make predictions on testing data
test_scaled = pd.DataFrame(test_scaled, columns=preprocessor.get_feature_names_out())
test_pred = best_model.predict(test_scaled)

# Save predictions to submission.csv
submission_df = pd.DataFrame({'id': test_df['id'], 'tested_positive_day3': test_pred})
submission_df.to_csv('/workspace/submission.csv', index=False)
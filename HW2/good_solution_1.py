# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the training data
train_df = pd.read_csv('/workspace/ML2025Spring-hw2-public/train.csv')

# Load the testing data
test_df = pd.read_csv('/workspace/ML2025Spring-hw2-public/test.csv')

# Define the preprocessing steps
numeric_features = ['cli_day1', 'ili_day1', 'wnohh_cmnty_cli_day1', 'wbelief_masking_effective_day1', 
                    'wbelief_distancing_effective_day1', 'wcovid_vaccinated_friends_day1', 
                    'wlarge_event_indoors_day1', 'wothers_masked_public_day1', 'wothers_distanced_public_day1', 
                    'wshop_indoors_day1', 'wrestaurant_indors_day1', 'wworried_catch_covid_day1', 
                    'hh_cmnty_cli_day1', 'nohh_cmnty_cli_day1', 'wearing_mask_7d_day1', 'public_transit_day1', 
                    'worried_finances_day1', 'tested_positive_day1', 'cli_day2', 'ili_day2', 'wnohh_cmnty_cli_day2', 
                    'wbelief_masking_effective_day2', 'wbelief_distancing_effective_day2', 'wcovid_vaccinated_friends_day2', 
                    'wlarge_event_indoors_day2', 'wothers_masked_public_day2', 'wothers_distanced_public_day2', 
                    'wshop_indoors_day2', 'wrestaurant_indoors_day2', 'wworried_catch_covid_day2', 
                    'hh_cmnty_cli_day2', 'nohh_cmnty_cli_day2', 'wearing_mask_7d_day2', 'public_transit_day2', 
                    'worried_finances_day2', 'tested_positive_day2', 'cli_day3', 'ili_day3', 'wnohh_cmnty_cli_day3', 
                    'wbelief_masking_effective_day3', 'wbelief_distancing_effective_day3', 'wcovid_vaccinated_friends_day3', 
                    'wlarge_event_indoors_day3', 'wothers_masked_public_day3', 'wothers_distanced_public_day3', 
                    'wshop_indoors_day3', 'wrestaurant_indoors_day3', 'wworried_catch_covid_day3', 
                    'hh_cmnty_cli_day3', 'nohh_cmnty_cli_day3', 'wearing_mask_7d_day3', 'public_transit_day3', 
                    'worried_finances_day3']

categorical_features = ['id', 'AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', RandomForestRegressor())])

# Define the hyperparameter tuning space
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]}

# Perform hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_df.drop('tested_positive_day3', axis=1), train_df['tested_positive_day3'])

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the testing data
predictions = best_model.predict(test_df)

# Save the predictions to a submission file
submission_df = pd.DataFrame(predictions, columns=['tested_positive_day3'])
submission_df['id'] = test_df['id']
submission_df.to_csv('/workspace/submission.csv', index=False)
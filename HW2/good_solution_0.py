import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Load the data
train_df = pd.read_csv('/workspace/ML2025Spring-hw2-public/train.csv')
test_df = pd.read_csv('/workspace/ML2025Spring-hw2-public/test.csv')

# Handle missing values
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Scale the features
scaler = StandardScaler()
train_df[['cli_day1', 'ili_day1', 'wnohh_cmnty_cli_day1', 'wbelief_masking_effective_day1',
         'wbelief_distancing_effective_day1', 'wcovid_vaccinated_friends_day1',
         'wlarge_event_indoors_day1', 'wothers_masked_public_day1', 'wothers_distanced_public_day1',
         'wshop_indoors_day1', 'wrestaurant_indoors_day1', 'wworried_catch_covid_day1',
         'hh_cmnty_cli_day1', 'nohh_cmnty_cli_day1', 'wearing_mask_7d_day1', 'public_transit_day1',
         'worried_finances_day1', 'tested_positive_day1', 'cli_day2', 'ili_day2',
         'wnohh_cmnty_cli_day2', 'wbelief_masking_effective_day2', 'wbelief_distancing_effective_day2',
         'wcovid_vaccinated_friends_day2', 'wlarge_event_indoors_day2', 'wothers_masked_public_day2',
         'wothers_distanced_public_day2', 'wshop_indoors_day2', 'wrestaurant_indoors_day2',
         'wworried_catch_covid_day2', 'hh_cmnty_cli_day2', 'nohh_cmnty_cli_day2',
         'wearing_mask_7d_day2', 'public_transit_day2', 'worried_finances_day2',
         'tested_positive_day2', 'cli_day3', 'ili_day3', 'wnohh_cmnty_cli_day3',
         'wbelief_masking_effective_day3', 'wbelief_distancing_effective_day3',
         'wcovid_vaccinated_friends_day3', 'wlarge_event_indoors_day3', 'wothers_masked_public_day3',
         'wothers_distanced_public_day3', 'wshop_indoors_day3', 'wrestaurant_indoors_day3',
         'wworried_catch_covid_day3', 'hh_cmnty_cli_day3', 'nohh_cmnty_cli_day3',
         'wearing_mask_7d_day3', 'public_transit_day3', 'worried_finances_day3']] = scaler.fit_transform(train_df[['cli_day1', 'ili_day1', 'wnohh_cmnty_cli_day1', 'wbelief_masking_effective_day1',
         'wbelief_distancing_effective_day1', 'wcovid_vaccinated_friends_day1',
         'wlarge_event_indoors_day1', 'wothers_masked_public_day1', 'wothers_distanced_public_day1',
         'wshop_indoors_day1', 'wrestaurant_indoors_day1', 'wworried_catch_covid_day1',
         'hh_cmnty_cli_day1', 'nohh_cmnty_cli_day1', 'wearing_mask_7d_day1', 'public_transit_day1',
         'worried_finances_day1', 'tested_positive_day1', 'cli_day2', 'ili_day2',
         'wnohh_cmnty_cli_day2', 'wbelief_masking_effective_day2', 'wbelief_distancing_effective_day2',
         'wcovid_vaccinated_friends_day2', 'wlarge_event_indoors_day2', 'wothers_masked_public_day2',
         'wothers_distanced_public_day2', 'wshop_indoors_day2', 'wrestaurant_indoors_day2',
         'wworried_catch_covid_day2', 'hh_cmnty_cli_day2', 'nohh_cmnty_cli_day2',
         'wearing_mask_7d_day2', 'public_transit_day2', 'worried_finances_day2',
         'tested_positive_day2', 'cli_day3', 'ili_day3', 'wnohh_cmnty_cli_day3',
         'wbelief_masking_effective_day3', 'wbelief_distancing_effective_day3',
         'wcovid_vaccinated_friends_day3', 'wlarge_event_indoors_day3', 'wothers_masked_public_day3',
         'wothers_distanced_public_day3', 'wshop_indoors_day3', 'wrestaurant_indoors_day3',
         'wworried_catch_covid_day3', 'hh_cmnty_cli_day3', 'nohh_cmnty_cli_day3',
         'wearing_mask_7d_day3', 'public_transit_day3', 'worried_finances_day3']])

test_df[['cli_day1', 'ili_day1', 'wnohh_cmnty_cli_day1', 'wbelief_masking_effective_day1',
         'wbelief_distancing_effective_day1', 'wcovid_vaccinated_friends_day1',
         'wlarge_event_indoors_day1', 'wothers_masked_public_day1', 'wothers_distanced_public_day1',
         'wshop_indoors_day1', 'wrestaurant_indoors_day1', 'wworried_catch_covid_day1',
         'hh_cmnty_cli_day1', 'nohh_cmnty_cli_day1', 'wearing_mask_7d_day1', 'public_transit_day1',
         'worried_finances_day1', 'tested_positive_day1', 'cli_day2', 'ili_day2',
         'wnohh_cmnty_cli_day2', 'wbelief_masking_effective_day2', 'wbelief_distancing_effective_day2',
         'wcovid_vaccinated_friends_day2', 'wlarge_event_indoors_day2', 'wothers_masked_public_day2',
         'wothers_distanced_public_day2', 'wshop_indoors_day2', 'wrestaurant_indoors_day2',
         'wworried_catch_covid_day2', 'hh_cmnty_cli_day2', 'nohh_cmnty_cli_day2',
         'wearing_mask_7d_day2', 'public_transit_day2', 'worried_finances_day2',
         'tested_positive_day2', 'cli_day3', 'ili_day3', 'wnohh_cmnty_cli_day3',
         'wbelief_masking_effective_day3', 'wbelief_distancing_effective_day3',
         'wcovid_vaccinated_friends_day3', 'wlarge_event_indoors_day3', 'wothers_masked_public_day3',
         'wothers_distanced_public_day3', 'wshop_indoors_day3', 'wrestaurant_indoors_day3',
         'wworried_catch_covid_day3', 'hh_cmnty_cli_day3', 'nohh_cmnty_cli_day3',
         'wearing_mask_7d_day3', 'public_transit_day3', 'worried_finances_day3']] = scaler.transform(test_df[['cli_day1', 'ili_day1', 'wnohh_cmnty_cli_day1', 'wbelief_masking_effective_day1',
         'wbelief_distancing_effective_day1', 'wcovid_vaccinated_friends_day1',
         'wlarge_event_indoors_day1', 'wothers_masked_public_day1', 'wothers_distanced_public_day1',
         'wshop_indoors_day1', 'wrestaurant_indoors_day1', 'wworried_catch_covid_day1',
         'hh_cmnty_cli_day1', 'nohh_cmnty_cli_day1', 'wearing_mask_7d_day1', 'public_transit_day1',
         'worried_finances_day1', 'tested_positive_day1', 'cli_day2', 'ili_day2',
         'wnohh_cmnty_cli_day2', 'wbelief_masking_effective_day2', 'wbelief_distancing_effective_day2',
         'wcovid_vaccinated_friends_day2', 'wlarge_event_indoors_day2', 'wothers_masked_public_day2',
         'wothers_distanced_public_day2', 'wshop_indoors_day2', 'wrestaurant_indoors_day2',
         'wworried_catch_covid_day2', 'hh_cmnty_cli_day2', 'nohh_cmnty_cli_day2',
         'wearing_mask_7d_day2', 'public_transit_day2', 'worried_finances_day2',
         'tested_positive_day2', 'cli_day3', 'ili_day3', 'wnohh_cmnty_cli_day3',
         'wbelief_masking_effective_day3', 'wbelief_distancing_effective_day3',
         'wcovid_vaccinated_friends_day3', 'wlarge_event_indoors_day3', 'wothers_masked_public_day3',
         'wothers_distanced_public_day3', 'wshop_indoors_day3', 'wrestaurant_indoors_day3',
         'wworried_catch_covid_day3', 'hh_cmnty_cli_day3', 'nohh_cmnty_cli_day3',
         'wearing_mask_7d_day3', 'public_transit_day3', 'worried_finances_day3']])

# Split the data into training and validation sets
splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splits.split(train_df, train_df['tested_positive_day3']):
    train_df_train, train_df_val = train_df.iloc[train_idx], train_df.iloc[val_idx]

# Define the model and hyperparameter tuning space
model = RandomForestRegressor(n_estimators=100, random_state=42)
param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 5, 10, 20]}

# Perform hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_df_train.drop('tested_positive_day3', axis=1), train_df_train['tested_positive_day3'])

# Train the model with the optimal hyperparameters
model = grid_search.best_estimator_
model.fit(train_df_train.drop('tested_positive_day3', axis=1), train_df_train['tested_positive_day3'])

# Make predictions on the validation set
val_pred = model.predict(train_df_val.drop('tested_positive_day3', axis=1))

# Evaluate the model on the validation set
val_mse = mean_squared_error(train_df_val['tested_positive_day3'], val_pred)
print(f'Validation MSE: {val_mse}')

# Make predictions on the testing set
test_pred = model.predict(test_df.drop('tested_positive_day3', axis=1))

# Save the predictions to a CSV file
submission_df = pd.DataFrame({'id': test_df['id'], 'tested_positive_day3': test_pred})
submission_df.to_csv('/workspace/submission.csv', index=False)
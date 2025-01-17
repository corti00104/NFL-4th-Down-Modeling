# -*- coding: utf-8 -*-
"""
Created on Fri May 31 18:41:01 2024

@author: NAWri
"""

""" Working on getting trendline to show up in time remaining probability and 
getting print functions for each of the binary measurables"""

#-------------------------------------------------------------------------------------
# ------------------- Data Loading, Organizing, and Manipulation ---------------------
#-------------------------------------------------------------------------------------
!pip install nfl_data_py

import nfl_data_py as nfl
import pandas as pd
import matplotlib as mpl

# Load play-by-play data for the 2021 to 2023 seasons
seasons = [2021, 2022, 2023]
pbp_data = nfl.import_pbp_data(seasons)

# Display the first few rows of the data
print(pbp_data.head())

stats_2021_data = pd.read_csv("C:/Users/natha/Documents/BGA/Summer2024/4th_Down_Prob/2021Stats.csv")
stats_2022_data = pd.read_csv("C:/Users/natha/Documents/BGA/4th_Down_Prob/2022Stats.csv")
stats_2023_data = pd.read_csv("C:/Users/NAWri/Documents/BGA/4th_Down_Prob/2023Stats.csv")


# Filter for 4th down plays using boolean indexing
data_4th_down = pbp_data[pbp_data['down'] == 4.0]

# Display the first few rows to verify
print(data_4th_down.head())

# Filter for 4th down conversion attempts
conversion_attempts = data_4th_down[data_4th_down['play_type'].isin(['run', 'pass'])]

"""
# FOR FUTURE USE (POSSIBLE DECISION TREE)
field_goal_attempts = data_4th_down[data_4th_down['play_type'].isin(['field_goal'])]
punt_decision = data_4th_down[data_4th_down['play_type'].isin(['punt'])]
"""

#Changing LA to LA Rams (LAR)
import re

# Iterate over each column in the dataframe
for column in conversion_attempts.columns:
    # Check if the column contains strings (dtype == 'object')
    if conversion_attempts[column].dtype == 'object':
        # Replace "LA" with "LAR" using regular expressions with word boundaries
        conversion_attempts[column] = conversion_attempts[column].apply(lambda x: re.sub(r'\bLA\b', 'LAR', x))
        
# List of columns to keep
columns_to_keep = ['posteam','posteam_type','defteam','side_of_field',
                   'yardline_100','quarter_seconds_remaining','half_seconds_remaining',
                   'game_seconds_remaining','qtr','down','ydstogo','ydsnet',
                   'yards_gained','epa','wp','wpa','vegas_wpa','pass_attempt',
                   'season','div_game','roof','surface',
                   'temp','wind','success','defenders_in_box','cp','cpoe']

# Drop columns not in the list of columns to keep
conversion_attempts = conversion_attempts[columns_to_keep]

# Add a 'season' column to each stats dataframe
stats_2021_data['season'] = 2021
stats_2022_data['season'] = 2022
stats_2023_data['season'] = 2023

# Combine the stats dataframes into a single dataframe
combined_stats = pd.concat([stats_2021_data, stats_2022_data, stats_2023_data], ignore_index=True)

# Separate offensive and defensive stats
offensive_stats = combined_stats[['Team', 'season', 'OffYPG', 'OffRank', 'PFFOL', 'OLRank']]
print (offensive_stats)
defensive_stats = combined_stats[['Team', 'season', 'DefYPG', 'DefRank', 'PFFDL', 'DLRank']]
print (defensive_stats)

# Rename columns to avoid conflicts
offensive_stats = offensive_stats.rename(columns={
    'Team': 'posteam',
    'OffYPG': 'off_OffYPG',
    'OffRank': 'off_OffRank',
    'PFFOL': 'off_PFFOL',
    'OLRank': 'off_OLRank'
})

defensive_stats = defensive_stats.rename(columns={
    'Team': 'defteam',
    'DefYPG': 'def_DefYPG',
    'DefRank': 'def_DefRank',
    'PFFDL': 'def_PFFDL',
    'DLRank': 'def_DLRank'
})

# Merge offensive stats with the conversion_attempts dataframe
conversion_attempts = pd.merge(
    conversion_attempts,
    offensive_stats,
    left_on=['posteam', 'season'],
    right_on=['posteam', 'season'],
    how='left'
)

# Merge defensive stats with the conversion_attempts dataframe
conversion_attempts = pd.merge(
    conversion_attempts,
    defensive_stats,
    left_on=['defteam', 'season'],
    right_on=['defteam', 'season'],
    how='left'
)

# Display the first few rows to verify
print(conversion_attempts.head())

#rename 'roof' to 'roof_open_air'
conversion_attempts = conversion_attempts.rename(columns ={
    'roof': 'roof_open_air'
})


# Modify the 'roof' variable to be binary with outdoors and open being 1, else being 0
conversion_attempts['roof_open_air'] = conversion_attempts['roof_open_air'].apply(lambda x: 1 if x in ['outdoors', 'open'] else 0)

# Verify the modification
print(conversion_attempts.head())

# change grass to be binary grass or turf
conversion_attempts = conversion_attempts.rename(columns ={
    'surface': 'grass'
})

conversion_attempts['grass'] = conversion_attempts['grass'].apply(lambda x: 1 if x in ['grass'] else 0)

# Combining rush and pass attempts. 1 being pass, 0 being run

conversion_attempts = conversion_attempts.rename(columns ={
    'pass_attempt' : 'pass_run'
})
# Deleting redundant columns
conversion_attempts = conversion_attempts.drop(columns=['side_of_field','posteam','defteam'])

# Changing posteam_type to Binary for home or away, changing to home_away

conversion_attempts['posteam_type'] = conversion_attempts['posteam_type'].apply(lambda x: 1 if x in ['home'] else 0)
conversion_attempts = conversion_attempts.rename(columns ={
    'posteam_type': 'posteam_home'
})

# Renaming yardline_100 to yards_to_endzone

conversion_attempts = conversion_attempts.rename(columns={
    'yardline_100':'yards_to_endzone'
})

# Renaming ydsnet to ydsondrive

conversion_attempts = conversion_attempts.rename(columns={
    'ydsnet': 'ydsondrive'
})

#----------------------------------------------------------------------------------
#---------------------------- Data Training and Testing ---------------------------
#----------------------------------------------------------------------------------

# List of potential independent variables
ind_vars = ['posteam_home','ydstogo', 'yards_to_endzone','qtr','ydsondrive','wp','pass_run',
                    'div_game','roof_open_air','grass','quarter_seconds_remaining',
                    'half_seconds_remaining','game_seconds_remaining',
                    'off_OffYPG', 'off_OffRank','def_DefYPG', 'def_DefRank',
                    'off_PFFOL', 'off_OLRank', 'def_PFFDL', 'def_DLRank',
                    'temp','wind','defenders_in_box','cp','cpoe',]

# Separate the independent and dependent variables
X = conversion_attempts[ind_vars]
Y = conversion_attempts['success']

import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import numpy as np

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Recreate a DataFrame with column names
X_imputed = pd.DataFrame(X_imputed, columns=ind_vars)

# Ensure that X and y have the same number of samples
print(f"Shape of X: {X.shape}")
print(f"Length of Y: {len(Y)}")

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, Y, test_size=0.2, random_state=42)

# Ensure that the training and test sets have the same number of samples
print(f"Shape of X_train: {X_train.shape}")
print(f"Length of Y_train: {len(Y_train)}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Length of Y_test: {len(Y_test)}")

# Check for NaN values in X_train and X_test
print("Number of NaN values in X_train:", np.isnan(X_train).sum())
print("Number of NaN values in X_test:", np.isnan(X_test).sum())

# Re-impute missing values if any are found
if np.isnan(X_train).sum() > 0 or np.isnan(X_test).sum() > 0:
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

# Fit the RandomForestClassifier after ensuring no NaNs are present
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Select the most important features
selector = SelectFromModel(rf, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train_selected, Y_train)

# Predictions and evaluation
Y_pred = log_reg.predict(X_test_selected)
accuracy = accuracy_score(Y_test, Y_pred)
roc_auc = roc_auc_score(Y_test, Y_pred)

print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)
print("Classification Report:\n", classification_report(Y_test, Y_pred))

cross_val_scores = cross_val_score(log_reg, X_train_selected, Y_train, cv=5, scoring='accuracy')
print("cross_val_scores", cross_val_scores)
print("Mean cross-validation accuracy:", cross_val_scores.mean())

#-------------------------------------------------------------------------------
#------------------------- Storing Results in Dataframes -----------------------
#-------------------------------------------------------------------------------

"Full Variable Set"
# Fit the logistic regression model on the original set of features
log_reg_full = LogisticRegression(max_iter=10000)
log_reg_full.fit(X_train, Y_train)

# Extract the coefficients
coefficients_full = log_reg_full.coef_[0]

# Create a DataFrame to store the coefficients and their corresponding feature names
coeff_df_full = pd.DataFrame({
    'Feature': ind_vars,
    'Coefficient': coefficients_full
})

# Display the DataFrame
print(coeff_df_full)

"Fitted Variable Set"

# Fit the logistic regression model on the training data
log_reg.fit(X_train_selected, Y_train)

# Extract the coefficients
coefficients = log_reg.coef_[0]

# Get the feature names from the selector
feature_names = np.array(ind_vars)[selector.get_support()]

# Create a DataFrame to store the coefficients and their corresponding feature names
coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Display the DataFrame
print(coeff_df)

#-------------------------------------------------------------------------------
# ------------------------ Data Plots and Visuals ------------------------------
#-------------------------------------------------------------------------------

# Success rate by distance -----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Group by 'ydstogo' and calculate the mean success rate
success_rates_by_yardage = conversion_attempts.groupby('ydstogo')['success'].mean()

# Filter for yard distances from 1 to 12
success_rates_by_yardage = success_rates_by_yardage.loc[1:12]

# Extract x and y values
x = success_rates_by_yardage.index
y = success_rates_by_yardage.values

# Fit a linear trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Plot the success rates
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', label='Success Rate')
plt.plot(x, p(x), "r--", label='Trend Line')  # Trend line

plt.title('Probability of Successful Conversion from 1 to 12 Yards Away')
plt.xlabel('Yards to Go')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True)
plt.show()

# Print percentages of conversion at each distance
print("Conversion Rates:")
for distance, success_rate in zip(x, y):
    print(f"Yards to Go: {distance}, Success Rate: {success_rate:.2%}")
print(f"Slope of the trend line: {z[0]}")


# Success Rates by Distance and Playtype ----------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Filter data for yardages from 1 to 12 yards
conversion_attempts_filtered = conversion_attempts[conversion_attempts['ydstogo'].between(1, 12)]

# Calculate conversion probabilities for each yardage for both passing and running
conversion_probabilities = conversion_attempts_filtered.groupby(['ydstogo', 'pass_run'])['success'].agg(['mean', 'count']).reset_index()

# Calculate overall conversion probabilities
overall_conversion_probabilities = conversion_attempts_filtered.groupby('ydstogo')['success'].agg(['mean', 'count']).reset_index()
overall_conversion_probabilities = overall_conversion_probabilities.rename(columns={'mean': 'overall_mean', 'count': 'overall_count'})

# Merge the overall conversion probabilities with the pass/run probabilities
conversion_probabilities = conversion_probabilities.merge(overall_conversion_probabilities, on='ydstogo')

# Filter out yardages with fewer than 10 conversion attempts
conversion_probabilities = conversion_probabilities[conversion_probabilities['count'] >= 10]
overall_conversion_probabilities = overall_conversion_probabilities[overall_conversion_probabilities['overall_count'] >= 10]

# Separate the pass and run conversion probabilities
pass_conversion_probabilities = conversion_probabilities[conversion_probabilities['pass_run'] == 1]
run_conversion_probabilities = conversion_probabilities[conversion_probabilities['pass_run'] == 0]

# Plot the results
plt.figure(figsize=(10, 6))

# Plot overall conversion probabilities
plt.plot(overall_conversion_probabilities['ydstogo'], overall_conversion_probabilities['overall_mean'], label='Overall Conversion Probability', color='black', marker='o')
# Fit and plot the trend line for overall conversion probabilities
z_overall = np.polyfit(overall_conversion_probabilities['ydstogo'], overall_conversion_probabilities['overall_mean'], 1)
p_overall = np.poly1d(z_overall)
plt.plot(overall_conversion_probabilities['ydstogo'], p_overall(overall_conversion_probabilities['ydstogo']), "k--")

# Plot pass conversion probabilities
plt.plot(pass_conversion_probabilities['ydstogo'], pass_conversion_probabilities['mean'], label='Pass Conversion Probability', color='blue', marker='o')
# Fit and plot the trend line for pass conversion probabilities
z_pass = np.polyfit(pass_conversion_probabilities['ydstogo'], pass_conversion_probabilities['mean'], 1)
p_pass = np.poly1d(z_pass)
plt.plot(pass_conversion_probabilities['ydstogo'], p_pass(pass_conversion_probabilities['ydstogo']), "b--")

# Plot run conversion probabilities
plt.plot(run_conversion_probabilities['ydstogo'], run_conversion_probabilities['mean'], label='Run Conversion Probability', color='red', marker='o')
# Fit and plot the trend line for run conversion probabilities
z_run = np.polyfit(run_conversion_probabilities['ydstogo'], run_conversion_probabilities['mean'], 1)
p_run = np.poly1d(z_run)
plt.plot(run_conversion_probabilities['ydstogo'], p_run(run_conversion_probabilities['ydstogo']), "r--")

# Add labels and legend
plt.xlabel('Yards to Go')
plt.ylabel('Conversion Probability')
plt.title('Conversion Probability by Play Type and Yards to Go')
plt.legend()

# Show plot
plt.show()

# Print conversion rates
print("Conversion Rates:")
for index, row in conversion_probabilities.iterrows():
    if row['pass_run'] == 1:
        print(f"Yards to Go: {row['ydstogo']}, Pass Success Rate: {row['mean']:.2%}")
    elif row['pass_run'] == 0:
        print(f"Yards to Go: {row['ydstogo']}, Run Success Rate: {row['mean']:.2%}")
    print(f"Yards to Go: {row['ydstogo']}, Overall Success Rate: {row['overall_mean']:.2%}")

# Print slopes of trend lines
print(f"Slope of overall trend line: {z_overall[0]}")
print(f"Slope of pass trend line: {z_pass[0]}")
print(f"Slope of run trend line: {z_run[0]}")


# Success Rates by Temperature ---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Group by temperature and calculate the mean success rate
success_rates_temp = conversion_attempts.groupby('temp')['success'].mean().reset_index()

# Extract x and y values
x = success_rates_temp['temp']
y = success_rates_temp['success']

# Fit a linear trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Plot the success rates by temperature
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', label='Success Rate')
plt.plot(x, p(x), "r--", label='Trend Line')  # Trend line

plt.title('Change in Conversion Probability with Temperature')
plt.xlabel('Temperature (F)')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True)
plt.show()
print(f"Slope of the trend line: {z[0]}")

# Success Rates by Time Remaining ------------------------------------------------
# Group by game_seconds_remaining and calculate the mean success rate
"Edit this graph to look nicer, lines too clustered together"
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Group by game half-minutes remaining and calculate the mean success rate
success_rates_time = conversion_attempts.groupby('game_halfminutes_remaining')['success'].mean()

# Define a quadratic function for curve fitting
def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the quadratic function to the grouped data points
popt, pcov = curve_fit(quadratic_func, success_rates_time.index, success_rates_time.values, p0=(1, 1, 1))  # Adjust p0 if needed

# Plot the success rates by game half-minutes remaining
plt.figure(figsize=(12, 8))
plt.plot(success_rates_time.index, success_rates_time.values, marker='o', linestyle='-', linewidth=2, label='Success Rate')
plt.plot(success_rates_time.index, quadratic_func(success_rates_time.index, *popt), color='red', linestyle='--', linewidth=2, label='Quadratic Trend Line')
plt.title('Change in Conversion Probability with Time Remaining (Half-Minutes)')
plt.xlabel('Game Half-Minutes Remaining')
plt.ylabel('Success Rate')
plt.grid(True)
plt.ylim(0, 1)
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.legend()
plt.tight_layout()
plt.show()

# Success Rates by Offensive Yardage per Game -------------------------------------

# Group by offensive YPG rank and calculate the mean success rate
success_rates_off_ypg_rank = conversion_attempts.groupby('off_OffRank')['success'].mean().reset_index()

# Extract x and y values
x = success_rates_off_ypg_rank['off_OffRank']
y = success_rates_off_ypg_rank['success']

# Fit a linear trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Plot the success rates by offensive YPG rank
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', label='Success Rate')
plt.plot(x, p(x), "r--", label='Trend Line')  # Trend line

plt.title('Change in Conversion Probability with Offensive YPG Rank')
plt.xlabel('Offensive YPG Rank')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True)
plt.show()
print(f"Slope of the trend line: {z[0]}")
# Success Rates by Offensive line rank ------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Group by offensive line rank and calculate the mean success rate
success_rates_ol_rank = conversion_attempts.groupby('off_OLRank')['success'].mean().reset_index()

# Extract x and y values
x = success_rates_ol_rank['off_OLRank']
y = success_rates_ol_rank['success']

# Fit a linear trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Plot the success rates by offensive line rank
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', label='Success Rate')
plt.plot(x, p(x), "r--", label='Trend Line')  # Trend line

plt.title('Change in Conversion Probability with Offensive Line Rank')
plt.xlabel('Offensive Line Rank')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True)
plt.show()
print(f"Slope of the trend line: {z[0]}")

# Success Rate by Opposing Defense rank ------------------------------------------

# Group by defensive rank and calculate the mean success rate
success_rates_def_rank = conversion_attempts.groupby('def_DefRank')['success'].mean().reset_index()

# Extract x and y values
x = success_rates_def_rank['def_DefRank']
y = success_rates_def_rank['success']

# Fit a linear trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Plot the success rates by defensive rank
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', label='Success Rate')
plt.plot(x, p(x), "r--", label='Trend Line')  # Trend line

plt.title('Change in Conversion Probability with Defensive Rank')
plt.xlabel('Defensive Rank')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True)
plt.show()
print(f"Slope of the trend line: {z[0]}")

# Success Rate by Opposing DL rank -------------------------------------------------

# Group by defensive line rank and calculate the mean success rate
success_rates_dl_rank = conversion_attempts.groupby('def_DLRank')['success'].mean().reset_index()

# Extract x and y values
x = success_rates_dl_rank['def_DLRank']
y = success_rates_dl_rank['success']

# Fit a linear trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Plot the success rates by defensive line rank
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', label='Success Rate')
plt.plot(x, p(x), "r--", label='Trend Line')  # Trend line

plt.title('Change in Conversion Probability with Defensive Line Rank')
plt.xlabel('Defensive Line Rank')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True)
plt.show()
print(f"Slope of the trend line: {z[0]}")


# Home vs Away split ----------------------------------------------------------------
# Calculate the success rates for home and away teams
success_rates_home_away = conversion_attempts.groupby('posteam_home')['success'].mean()

# Plot the success rates
success_rates_home_away.plot(kind='bar', figsize=(8, 5))
plt.title('Success Rate: Home vs. Away')
plt.xlabel('Home (1) vs. Away (0)')
plt.ylabel('Success Rate')
plt.xticks(ticks=[0, 1], labels=['Away', 'Home'], rotation=0)
plt.grid(True)
plt.show()

# Print percentages
print("Success Rate - Home Team:", success_rates_home_away[1])
print("Success Rate - Away Team:", success_rates_home_away[0])

# Divisional Game split -------------------------------------------------------------
# Calculate the success rates for divisional and non-divisional games
success_rates_div_game = conversion_attempts.groupby('div_game')['success'].mean()

# Plot the success rates
success_rates_div_game.plot(kind='bar', figsize=(8, 5))
plt.title('Success Rate: Divisional vs. Non-divisional Games')
plt.xlabel('Divisional Game (1) vs. Non-divisional Game (0)')
plt.ylabel('Success Rate')
plt.xticks(ticks=[0, 1], labels=['Non-divisional', 'Divisional'], rotation=0)
plt.grid(True)
plt.show()
print("Success Rate - Non-divisional Game:", success_rates_div_game[0])
print("Success Rate - Divisional Game:", success_rates_div_game[1])

# "indoor"/outdoor split --------------------------------------------------------------
# Calculate the success rates for games with and without a roof
success_rates_roof = conversion_attempts.groupby('roof_open_air')['success'].mean()

# Plot the success rates
success_rates_roof.plot(kind='bar', figsize=(8, 5))
plt.title('Success Rate: Roof (Open/Outdoors) vs. No Roof (Closed/Indoors)')
plt.xlabel('Roof (1) vs. No Roof (0)')
plt.ylabel('Success Rate')
plt.xticks(ticks=[0, 1], labels=['No Roof', 'Roof'], rotation=0)
plt.grid(True)
plt.show()
print("Success Rate - No Roof:", success_rates_roof[0])
print("Success Rate - Roof:", success_rates_roof[1])

# Grass v Turf split ------------------------------------------------------------------
# Calculate the success rates for games played on turf vs. grass
success_rates_surface = conversion_attempts.groupby('grass')['success'].mean()

# Plot the success rates
success_rates_surface.plot(kind='bar', figsize=(8, 5))
plt.title('Success Rate: Grass vs. Turf')
plt.xlabel('Grass (1) vs. Turf (0)')
plt.ylabel('Success Rate')
plt.xticks(ticks=[0, 1], labels=['Turf', 'Grass'], rotation=0)
plt.grid(True)
plt.show()

print("Success Rate - Turf:", success_rates_surface[0])
print("Success Rate - Grass:", success_rates_surface[1])

#-------------------------------------------------------------------------------
# ------------------------ Input Situation for probability ---------------------
#-------------------------------------------------------------------------------

# Function to calculate the averages of the independent variables for the specified teams
def get_team_averages(off_team, def_team, dataset):
    team_data = dataset[(dataset['off_OffRank'] == off_team) & (dataset['def_DefRank'] == def_team)]
    averages = {}
    for var in ind_vars:
        if var in team_data.columns:
            averages[var] = team_data[var].mean()
    return averages

# Function to estimate the conversion probability
def estimate_conversion_probability(yards_to_go, off_team, def_team, dataset, model, selector, scaler):
    averages = get_team_averages(off_team, def_team, dataset)
    
    # Update averages with the provided yards to go
    averages['ydstogo'] = yards_to_go
    
    # Prepare the input data for the model
    input_data = pd.DataFrame([averages])
    
    # Impute missing values
    input_data = imputer.transform(input_data)
    
    # Standardize the input data
    input_data = scaler.transform(input_data)
    
    # Select important features
    input_data_selected = selector.transform(input_data)
    
    # Estimate the conversion probability
    conversion_probability = model.predict_proba(input_data_selected)[:, 1][0]
    
    return conversion_probability

# Example usage
yards_to_go = 2
off_team = 'JAX'  # Replace with actual offensive team rank
def_team = 'TEN'  # Replace with actual defensive team rank

conversion_probability = estimate_conversion_probability(yards_to_go, off_team, def_team, conversion_attempts, log_reg, selector, scaler)
print(f"Estimated Conversion Probability: {conversion_probability:.2%}")
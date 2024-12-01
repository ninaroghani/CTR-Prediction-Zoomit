import openpyxl as openpyxl
from ydata_profiling import ProfileReport
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from hazm import stopwords_list
from sklearn.preprocessing import StandardScaler


####################### Data Preprocess ###############################
df = pd.read_excel("DataSet.xlsx", engine='openpyxl')
df = df[df["url_Url Clicks"]>10]
df = df[df["url_Impressions"]>500]
df.info()

# Generate a profile report
profile = ProfileReport(df)


df['Month'] = df['Month'].replace({'Farvardin':1,'Ordibehesht':2,'Khordad':3,'Tir':4,'Mordad':5,
                                                     'Shahrivar':6,'Mehr':7,'Aban':8,'Azar':9,'Dey':10,'Bahman':11,'Esfand':12})

df.sort_values(by='Month', ascending=True)

df.set_index('Column1',inplace=True)

df.drop(['Published Date','Comment Count', 'Like Count', 'zoomit_page_view', 'zoomit_page_view.1', 'zoomit_page_view.2',
         'zoomit_page_view.3','zoomit_ad_click','zoomit_ad_click.1', 'zoomit_ad_click.2', 'zoomit_ad_close',
         'zoomit_ad_close.1', 'zoomit_ad_close.2', 'zoomit_Totals', 'Platform',  'discover_Url Clicks',
         'discover_URL CTR', 'Team', 'Year_x', 'Year_y', 'Categories', 'url_Url Clicks',
         ] ,axis=1,inplace=True)

# Drop duplicates by 'TopicID', keeping only the first occurrence
df = df.drop_duplicates(subset='TopicID', keep='first')

######## out_of_range values Handeling ##########
df['Word Count'] = df['Word Count'].apply(lambda x: x if 200 <= x <= 5000 else np.nan)
df['Word Count'].min()
df['Word Count'].max()

df['Reading Time'] = df['Reading Time'].apply(lambda x: x if 1 <= x <= 25 else np.nan)
df['Reading Time'].min()
df['Reading Time'].max()

plt.figure(figsize=(5, 3))
plt.scatter(df['Word Count'], df['Reading Time'])
plt.xlabel('Word Count')
plt.ylabel('Reading Time')
plt.show()

correlation = df['Word Count'].corr(df['Reading Time'], method = 'pearson')
print("correlation",correlation)

######### Mising Values Imputation ###########
# Create a copy of the DataFrame to store imputed values
df_for_imputation = df[['Word Count', 'Reading Time']]

# Create IterativeImputer instance for 'Word Count' column
WordCount_imputer = IterativeImputer()

# Impute missing values in 'Word Count' column
df[['Word Count']] = WordCount_imputer.fit_transform(df_for_imputation[['Word Count']])

df.drop('Reading Time', axis=1, inplace=True)

################ Outlier Checking ###################
import pandas as pd

outlier_list = ['Word Count', 'url_Impressions', 'url_Average Position']

# Iterate over each column in the DataFrame
for column in df[outlier_list].columns:
    # Extract the column data
    data = df[column]

    # calculate interquartile range
    q25, q75 = data.quantile(0.25), data.quantile(0.75)
    iqr = q75 - q25

    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    globals()[f'lower_{column}_iqr'] = lower
    globals()[f'upper_{column}_iqr'] = upper

    # identify outliers
    globals()[f'outliers_{column}_iqr'] = [x for x in data if x < lower or x > upper]

    print(f"Column: {column}")
    print(f"Identified outliers: {len(globals()[f'outliers_{column}_iqr'])}")
    globals()[f'outliers_{column}_iqr'].sort()
    print('outlier values: ', globals()[f'outliers_{column}_iqr'])
    print("\n")

#########Checking Distributions ############
plt.figure(figsize = (6,4))
plt.hist(df['Word Count'])
plt.xlabel('Word Count')
plt.show()

plt.figure(figsize = (6,4))
plt.hist(df['url_Impressions'])
plt.xlabel('url_Impressions')
plt.show()

plt.figure(figsize = (6,4))
plt.hist(df['url_Average Position'])
plt.xlabel('url_Average Position')
plt.show()

plt.figure(figsize = (6,4))
plt.hist(df['url_URL CTR'])
plt.xlabel('url_URL CTR')
plt.show()

########## Feature Transformation ################
WordCount_transformer = PowerTransformer(method='box-cox', standardize = False)
df[['Word Count']] = WordCount_transformer.fit_transform(df[['Word Count']])

CTR_transformer = PowerTransformer(method='box-cox', standardize = False)
df[['url_Impressions']] = CTR_transformer.fit_transform(df[['url_Impressions']])

df[['url_URL CTR']] = CTR_transformer.fit_transform(df[['url_URL CTR']])

df[['url_Average Position']] = CTR_transformer.fit_transform(df[['url_Average Position']])

############ TF_IDF Transformation ###########
# Load Persian stop words from hazm
persian_stop_words = list(set(stopwords_list()))

# Initialize TfidfVectorizers with Persian stop words
vectorizer_slug = TfidfVectorizer(stop_words=persian_stop_words)
vectorizer_author = TfidfVectorizer(stop_words=persian_stop_words)
vectorizer_title = TfidfVectorizer(stop_words=persian_stop_words)

# Fit and transform each column independently
tfidf_slug = vectorizer_slug.fit_transform(df['Slug'])
tfidf_author = vectorizer_author.fit_transform(df['Author Name'])
tfidf_title = vectorizer_title.fit_transform(df['Title'])

# Concatenate the separate TF-IDF matrices
tfidf_combined = hstack([tfidf_slug, tfidf_author, tfidf_title])

# Optional: Convert to DataFrame for a viewable structure
df_word_embedding = pd.DataFrame(tfidf_combined.toarray(),
                                 columns=(list(vectorizer_slug.get_feature_names_out()) +
                                          list(vectorizer_author.get_feature_names_out()) +
                                          list(vectorizer_title.get_feature_names_out())))

# Count non-zero values in each column
non_zero_counts = df_word_embedding.astype(bool).sum(axis=0)

# Filter columns with more than 5 non-zero value
df_screened = df_word_embedding.loc[:, non_zero_counts > 5]

# Select the additional columns
additional_columns = df[['Month', 'Word Count', 'Video', 'url_Impressions','url_URL CTR','url_Average Position']]

# Concatenate the TF-IDF features with the selected additional columns
df_combined = pd.concat([df_screened, additional_columns.reset_index(drop=True)], axis=1)

############ Feature Encodeing #############
df_combined['Video'] = df_combined['Video'].replace({False:0,True:1})

########### Feature Scaling #############
inputs = df_combined.drop(columns=['url_URL CTR'])
y = df_combined['url_URL CTR']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(inputs)
X = pd.DataFrame(X_scaled, columns=inputs.columns)

############################EDA #################################
plt.figure(figsize = (6,4))
plt.hist(df['Word Count'])
plt.xlabel('Word Count')
plt.show()

plt.figure(figsize = (6,4))
plt.hist(df['url_Impressions'])
plt.xlabel('url_Impressions')
plt.show()

plt.figure(figsize = (6,4))
plt.hist(df['url_Average Position'])
plt.xlabel('url_Average Position')
plt.show()

plt.figure(figsize = (6,4))
plt.hist(df['url_URL CTR'])
plt.xlabel('url_URL CTR')
plt.show()

##################################################################
# Calculate the frequency of each month
month_counts = df['Month'].value_counts().sort_index()

# Create a bar plot with the height proportional to the frequency of each month
plt.figure(figsize=(6, 4))
month_counts.plot(kind='bar')

# Add labels and title
plt.xlabel('Month')
plt.ylabel('Frequency')


# Display the plot
plt.xticks(ticks=range(12), labels=range(1, 13), rotation=0)
plt.show()

#################################################
# Calculate the frequency
month_counts = df['Video'].value_counts().sort_index()

# Create a bar plot with the height proportional to the frequency
plt.figure(figsize=(6, 4))
month_counts.plot(kind='bar')

# Add labels and title
plt.xlabel('Video')
plt.ylabel('Frequency')

plt.show()
###################################################################
# Select the relevant columns from the dataframe
columns = ['Month', 'Word Count', 'Video', 'url_Impressions', 'url_Average Position', 'url_URL CTR']

# Calculate the correlation matrix
correlation_matrix = df[columns].corr()

# Create the heatmap
plt.figure(figsize=(12.5, 10.5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)

plt.xticks(rotation=30, ha='right', fontsize=14)
plt.yticks(rotation=30, ha='right', fontsize=14)

# Display the plot
plt.show()

###########################################################
# Specify the columns to calculate the correlation
columns_to_correlate = ['Month', 'Word Count', 'Video', 'url_Impressions', 'url_Average Position']

# Calculate Pearson correlation between each specified column and 'url_URL CTR'
correlation_results = df[columns_to_correlate + ['url_URL CTR']].corr(method='pearson')['url_URL CTR']

# Drop 'url_URL CTR' from the results as it's always 1 with itself
correlation_results = correlation_results.drop('url_URL CTR')

# Display the correlation results
print(correlation_results)

######################### Data Modeling ############################
#################LR Model ##################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set up nested cross-validation
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# To store evaluation metrics
outer_mse = []
outer_mae = []

for train_idx, test_idx in outer_cv.split(X, y):
    # Split data into outer training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Initialize the Linear Regression model
    linear_model = LinearRegression()

    # Train the model on the training set
    linear_model.fit(X_train, y_train)

    # Predict on the outer test set
    y_pred = linear_model.predict(X_test)

    # Inverse transform predictions and true values
    y_pred_original = CTR_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = CTR_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()

    nan_indices_test = np.where(np.isnan(y_test_original))[0]
    nan_indices_pred = np.where(np.isnan(y_pred_original))[0]

    # Combine indices from both arrays
    nan_indices = np.unique(np.concatenate((nan_indices_test, nan_indices_pred)))

    # Drop NaN indices from both arrays
    y_test_cleaned = np.delete(y_test_original, nan_indices)
    y_pred_cleaned = np.delete(y_pred_original, nan_indices)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_cleaned, y_pred_cleaned)
    mae = mean_absolute_error(y_test_cleaned, y_pred_cleaned)

    # Store metrics
    outer_mse.append(mse)
    outer_mae.append(mae)

# Final Results
print(f"Average MSE across folds: {np.mean(outer_mse):.4f}")
print(f"Average MAE across folds: {np.mean(outer_mae):.4f}")

###############XGBoost Model ################
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Set up nested cross-validation
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# To store evaluation metrics
outer_mse = []
outer_mae = []

for train_idx, test_idx in outer_cv.split(X, y):
    # Split data into outer training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Rename columns to generic names for XGBoost
    X_train.columns = [f"feature_{i}" for i in range(X.shape[1])]
    X_test.columns = [f"feature_{i}" for i in range(X.shape[1])]

    # Inner loop for hyperparameter tuning using GridSearchCV
    xgb_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=inner_cv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Use the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Predict on the outer test set
    y_pred = best_model.predict(X_test)

    # Inverse transform predictions and true values
    y_pred_original = CTR_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = CTR_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)

    # Store metrics
    outer_mse.append(mse)
    outer_mae.append(mae)

# Final Results
print(f"Average MSE across folds: {np.mean(outer_mse):.4f}")
print(f"Average MAE across folds: {np.mean(outer_mae):.4f}")

################## Random Forest Model #############
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}

# Set up nested cross-validation
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# To store evaluation metrics
outer_mse = []
outer_mae = []

for train_idx, test_idx in outer_cv.split(X, y):
    # Split data into outer training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Inner loop for hyperparameter tuning using GridSearchCV
    rf_model = RandomForestRegressor(max_features='sqrt', random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=inner_cv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Use the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Predict on the outer test set
    y_pred = best_model.predict(X_test)

    # Inverse transform predictions and true values
    y_pred_original = CTR_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = CTR_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)

    # Store metrics
    outer_mse.append(mse)
    outer_mae.append(mae)

# Final Results
print(f"Average MSE across folds: {np.mean(outer_mse):.4f}")
print(f"Average MAE across folds: {np.mean(outer_mae):.4f}")

################ SVR Model ##################
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2]
}

# Set up nested cross-validation
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# To store evaluation metrics
outer_mse = []
outer_mae = []

for train_idx, test_idx in outer_cv.split(X, y):
    # Split data into outer training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Inner loop for hyperparameter tuning using GridSearchCV
    svr_model = SVR()
    grid_search = GridSearchCV(
        estimator=svr_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=inner_cv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Use the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Predict on the outer test set
    y_pred = best_model.predict(X_test)

    # Inverse transform predictions and true values
    y_pred_original = CTR_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = CTR_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)

    # Store metrics
    outer_mse.append(mse)
    outer_mae.append(mae)

# Final Results
print(f"Average MSE across folds: {np.mean(outer_mse):.4f}")
print(f"Average MAE across folds: {np.mean(outer_mae):.4f}")

############ LASSO Regression Model ################
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Set up nested cross-validation
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# To store evaluation metrics
outer_mse = []
outer_mae = []

for train_idx, test_idx in outer_cv.split(X, y):
    # Split data into outer training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Inner loop for hyperparameter tuning using GridSearchCV
    lasso_model = Lasso(random_state=42, max_iter=10000)
    grid_search = GridSearchCV(
        estimator=lasso_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=inner_cv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Use the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Predict on the outer test set
    y_pred = best_model.predict(X_test)

    # Inverse transform predictions and true values
    y_pred_original = CTR_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = CTR_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)

    # Store metrics
    outer_mse.append(mse)
    outer_mae.append(mae)

# Final Results
print(f"Average MSE across folds: {np.mean(outer_mse):.4f}")
print(f"Average MAE across folds: {np.mean(outer_mae):.4f}")

####################### Model Performance Comparison ############################
# Model names and corresponding performance metrics
models = ['XGBoost', 'Random Forest', 'SVR', 'LASSO']
mse_values = [0.0018, 0.0019, 0.0019, 0.0019]
mae_values = [0.0261, 0.0264, 0.0278, 0.0271]

# Positioning the bars
x = np.arange(len(models))
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(6, 5))

# Plot MSE and MAE
rects1 = ax.bar(x - width/2, mse_values, width, label='MSE', color='skyblue')
rects2 = ax.bar(x + width/2, mae_values, width, label='MAE', color='salmon')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Performance Metrics', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend()

# Adding value annotations on bars
def add_values(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_values(rects1)
add_values(rects2)

# Show plot
plt.tight_layout()
plt.show()

######################## Feature Importance ######################
import shap
import xgboost as xgb

# Ensure feature names are unique by appending a suffix to duplicates
unique_feature_names = pd.Index(X.columns).to_series().duplicated(keep=False)
X.columns = [f"{col}_{i}" if is_duplicate else col
             for i, (col, is_duplicate) in enumerate(zip(X.columns, unique_feature_names))]

# Initialize the XGBoost Regressor with specified parameters
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)

# Train the model on the entire dataset
xgb_regressor.fit(X, y)

# Use SHAP to explain the model predictions
explainer = shap.Explainer(xgb_regressor, X)
shap_values = explainer(X)

# Compute mean absolute SHAP values for each feature
shap_feature_importance = np.abs(shap_values.values).mean(axis=0)

# Create a DataFrame for SHAP feature importance
shap_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': shap_feature_importance
}).sort_values(by='Importance', ascending=False)

# Select the top 10 features
top_10_features = shap_importance_df.head(10)
new_feature_names = ["url_Impressions", 'Mobile','url_Average Position', 'Google','Month', 'Word Count',
                    'YouTube', 'Phone','OS','Ahmadi']
top_10_features['Feature'] = new_feature_names

# Plot SHAP feature importances for the top 10 features
plt.figure(figsize=(12, 5))
plt.barh(top_10_features['Feature'], top_10_features['Importance'], color='skyblue')
plt.xlabel("SHAP Importance")
plt.ylabel("Feature")
plt.title("Top 10 SHAP Feature Importances")
plt.gca().invert_yaxis()
plt.show()

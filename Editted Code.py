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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from hazm import stopwords_list, Normalizer

####################### Data Preprocess ###############################
df = pd.read_excel("DataSet.xlsx", engine='openpyxl')
df = df[df["url_Url Clicks"]>10]
df = df[df["url_Impressions"]>500]

# Generate a profile report
profile = ProfileReport(df)


df['Month'] = df['Month'].replace({'Farvardin':1,'Ordibehesht':2,'Khordad':3,'Tir':4,'Mordad':5,
                                                     'Shahrivar':6,'Mehr':7,'Aban':8,'Azar':9,'Dey':10,'Bahman':11,'Esfand':12})

#df.sort_values(by='Month', ascending=True)

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
df['Word Count'].min()
df['Word Count'].max()

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

df['Video'] = df['Video'].replace({False:0,True:1})

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
# Load Persian stop words from hazm
persian_stop_words = list(set(stopwords_list()))

# Define TF-IDF vectorizers globally (will be re-fitted inside the inner loop)
vectorizer_slug = TfidfVectorizer(stop_words=persian_stop_words)
vectorizer_author = TfidfVectorizer(stop_words=persian_stop_words)
vectorizer_title = TfidfVectorizer(stop_words=persian_stop_words)

# Define hyperparameter grids for all models
param_grids = {
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "Random Forest": {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    "SVR": {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2]
    },
    "Lasso": {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }
}

models = {
    "XGBoost": XGBRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(max_features='sqrt', random_state=42),
    "SVR": SVR(),
    "Lasso": Lasso(random_state=42, max_iter=10000)
}

# Set up nested cross-validation
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Loop through models
results = {}
for model_name, model in models.items():
    outer_mse = []
    outer_mae = []

    print(f"Running nested cross-validation for {model_name}...")

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(df)):
        print('i=', i)

        # Split data into outer training and testing sets
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        # Inner loop for hyperparameter tuning
        best_mse = float('inf')
        best_model = None

        for j, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_df)):
            print('j=', j)

            # Split inner training and validation data
            inner_train_df, inner_val_df = train_df.iloc[inner_train_idx], train_df.iloc[inner_val_idx]

            ########## Inner Data Preparation ##########
            # Fit and transform TF-IDF on the inner training data
            tfidf_slug = vectorizer_slug.fit_transform(inner_train_df['Slug'])
            tfidf_author = vectorizer_author.fit_transform(inner_train_df['Author Name'])
            tfidf_title = vectorizer_title.fit_transform(inner_train_df['Title'])

            # Transform inner validation data using the fitted TF-IDF vectorizers
            tfidf_slug_val = vectorizer_slug.transform(inner_val_df['Slug'])
            tfidf_author_val = vectorizer_author.transform(inner_val_df['Author Name'])
            tfidf_title_val = vectorizer_title.transform(inner_val_df['Title'])

            # Combine TF-IDF matrices
            train_tfidf_combined = hstack([tfidf_slug, tfidf_author, tfidf_title])
            val_tfidf_combined = hstack([tfidf_slug_val, tfidf_author_val, tfidf_title_val])

            # Convert to DataFrame for filtering columns
            train_df_word_embedding = pd.DataFrame(train_tfidf_combined.toarray(),
                                                   columns=(list(vectorizer_slug.get_feature_names_out()) +
                                                            list(vectorizer_author.get_feature_names_out()) +
                                                            list(vectorizer_title.get_feature_names_out())))
            val_df_word_embedding = pd.DataFrame(val_tfidf_combined.toarray(),
                                                 columns=(list(vectorizer_slug.get_feature_names_out()) +
                                                          list(vectorizer_author.get_feature_names_out()) +
                                                          list(vectorizer_title.get_feature_names_out())))

            # Count non-zero values and filter columns
            non_zero_counts = train_df_word_embedding.astype(bool).sum(axis=0)
            selected_columns = pd.Index(non_zero_counts[non_zero_counts > 5].index).unique()

            # Filter the training data with selected columns
            train_df_screened = train_df_word_embedding[selected_columns]

            # Align validation data with the selected columns from training data
            val_df_screened = val_df_word_embedding[selected_columns]

            # Select additional columns
            additional_columns_train = inner_train_df[['Month', 'Word Count', 'Video', 'url_Impressions',
                                                       'url_Average Position']].reset_index(drop=True)
            additional_columns_val = inner_val_df[['Month', 'Word Count', 'Video', 'url_Impressions',
                                                   'url_Average Position']].reset_index(drop=True)

            # Combine screened TF-IDF features with additional columns
            train_combined = pd.concat([train_df_screened, additional_columns_train], axis=1)
            val_combined = pd.concat([val_df_screened, additional_columns_val], axis=1)

            ########## Feature Scaling ##########
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_combined)
            X_val_scaled = scaler.transform(val_combined)

            y_train = inner_train_df['url_URL CTR']
            y_val = inner_val_df['url_URL CTR']

            ########## Model Training and Validation ##########
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                scoring='neg_mean_squared_error',
                cv=None,
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)

            # Evaluate on inner validation set
            y_val_pred = grid_search.best_estimator_.predict(X_val_scaled)

            y_val_pred_original = CTR_transformer.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
            y_val_original = CTR_transformer.inverse_transform(y_val.to_numpy().reshape(-1, 1)).flatten()

            val_mse = mean_squared_error(y_val_original, y_val_pred_original)

            if val_mse < best_mse:
                best_mse = val_mse
                best_param = grid_search.best_params_

            print(X_train_scaled.shape, X_val_scaled.shape)
        print(best_param)

        ########## Outer Evaluation ##########
        # Repeat TF-IDF transformation and scaling for outer train and test data
        tfidf_slug_outer = vectorizer_slug.fit_transform(train_df['Slug'])
        tfidf_author_outer = vectorizer_author.fit_transform(train_df['Author Name'])
        tfidf_title_outer = vectorizer_title.fit_transform(train_df['Title'])

        tfidf_slug_test = vectorizer_slug.transform(test_df['Slug'])
        tfidf_author_test = vectorizer_author.transform(test_df['Author Name'])
        tfidf_title_test = vectorizer_title.transform(test_df['Title'])

        # Combine TF-IDF matrices
        train_tfidf_combined_outer = hstack([tfidf_slug_outer, tfidf_author_outer, tfidf_title_outer])
        test_tfidf_combined = hstack([tfidf_slug_test, tfidf_author_test, tfidf_title_test])

        # Convert to DataFrame for filtering columns
        train_df_word_embedding_outer = pd.DataFrame(train_tfidf_combined_outer.toarray(),
                                                     columns=(list(vectorizer_slug.get_feature_names_out()) +
                                                              list(vectorizer_author.get_feature_names_out()) +
                                                              list(vectorizer_title.get_feature_names_out())))
        test_df_word_embedding = pd.DataFrame(test_tfidf_combined.toarray(),
                                              columns=(list(vectorizer_slug.get_feature_names_out()) +
                                                       list(vectorizer_author.get_feature_names_out()) +
                                                       list(vectorizer_title.get_feature_names_out())))

        # Count non-zero values and filter columns
        non_zero_counts_outer = train_df_word_embedding_outer.astype(bool).sum(axis=0)
        selected_columns_outer = pd.Index(non_zero_counts_outer[non_zero_counts_outer > 5].index).unique()

        # Filter the training data with selected columns
        train_df_screened_outer = train_df_word_embedding_outer[selected_columns_outer]

        # Align validation data with the selected columns from training data
        test_df_screened = test_df_word_embedding[selected_columns_outer]

        # Select additional columns
        additional_columns_train_outer = train_df[['Month', 'Word Count', 'Video', 'url_Impressions',
                                                   'url_Average Position']].reset_index(drop=True)
        additional_columns_test = test_df[['Month', 'Word Count', 'Video', 'url_Impressions',
                                           'url_Average Position']].reset_index(drop=True)

        # Combine screened TF-IDF features with additional columns
        train_combined_outer = pd.concat([train_df_screened_outer, additional_columns_train_outer], axis=1)
        test_combined = pd.concat([test_df_screened, additional_columns_test], axis=1)

        ########## Feature Scaling ##########
        scaler = StandardScaler()
        X_train_scaled_outer = scaler.fit_transform(train_combined_outer)
        X_test_scaled = scaler.transform(test_combined)

        y_train_outer = train_df['url_URL CTR']
        y_test = test_df['url_URL CTR']

        ########## Model Training and Evaluation ##########

        outer_model = model.set_params(**grid_search.best_params_)
        outer_model.fit(X_train_scaled_outer, y_train_outer)

        print(X_test_scaled.shape)

        # Predict and evaluate
        y_test_pred = outer_model.predict(X_test_scaled)

        # Inverse transform predictions and true values
        y_pred_test_original = CTR_transformer.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
        y_test_original = CTR_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()

        test_mse = mean_squared_error(y_test_original, y_pred_test_original)
        test_mae = mean_absolute_error(y_test_original, y_pred_test_original)

        outer_mse.append(test_mse)
        outer_mae.append(test_mae)

    # Store results for the model
    results[model_name] = {
        "Average MSE": np.mean(outer_mse),
        "Average MAE": np.mean(outer_mae)
    }
    print(f"{model_name} - Average MSE: {np.mean(outer_mse):.4f}, Average MAE: {np.mean(outer_mae):.4f}")

# Final Results for All Models
print("Final Results:")
for model_name, metrics in results.items():
    print(f"{model_name} - MSE: {metrics['Average MSE']:.4f}, MAE: {metrics['Average MAE']:.4f}")

####################### Model Performance Comparison ############################
# Model names and corresponding performance metrics
models = ['XGBoost', 'Random Forest', 'SVR', 'LASSO']
mse_values = [0.0018, 0.0019, 0.0020, 0.0019]
mae_values = [0.0265, 0.0265, 0.0279, 0.0274]

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

########### Feature Scaling #############
inputs = df_combined.drop(columns=['url_URL CTR'])
y = df_combined['url_URL CTR']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(inputs)
X = pd.DataFrame(X_scaled, columns=inputs.columns)

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

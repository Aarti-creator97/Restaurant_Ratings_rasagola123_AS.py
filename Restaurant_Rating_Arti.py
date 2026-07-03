import pandas as pd
import numpy as np
from pathlib import Path

# Step 1: Load the data (change path if needed)
dataset_path = Path(r"C:\Users\Admin\Downloads\Arati_Congnifyz_internship\Dataset .csv")
dataset_df = pd.read_csv(dataset_path)

# Target column we want to predict
target_rating = "Aggregate rating"

# These two look like they directly tell you the rating, so we'll drop them
excluded_columns = ["Rating color", "Rating text"]

feature_data = dataset_df.drop(columns=[target_rating] + excluded_columns)
target_data = dataset_df[target_rating]


# Step 2: Split into train and test sets (80/20)
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    feature_data, target_data, test_size=0.2, random_state=42
)


# Step 3: Set up preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Separate columns into numeric and categorical (some bools might sneak in)
numeric_columns = train_data.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = train_data.select_dtypes(exclude=["number", "bool"]).columns.tolist()

# Probably safe to fill missing numbers with median
numeric_pipeline = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

# Fill missing categories with mode (most common value), then one-hot encode
categorical_pipeline = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing for all columns
data_processor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# Step 4: Train Gradient Boosting model
from sklearn.ensemble import GradientBoostingRegressor

# Using basic GradientBoosting — haven’t tuned params yet
gradient_model = GradientBoostingRegressor(random_state=42)

# Full pipeline: preprocess → model
model_pipeline = Pipeline(steps=[
    ("preprocessing", data_processor),
    ("regressor", gradient_model)
])

# Fit everything on training data
model_pipeline.fit(train_data, train_target)


# Step 5: Evaluate model on the test set
from sklearn.metrics import mean_squared_error, r2_score

predicted_values = model_pipeline.predict(test_data)

rmse_score = np.sqrt(mean_squared_error(test_target, predicted_values))
r2_score_value = r2_score(test_target, predicted_values)

print("\nModel Performance on Test Set:")
print(f" - RMSE: {rmse_score:.3f}")
print(f" - R² Score: {r2_score_value:.3f}")


# Step 6: Interpret using permutation importance
from sklearn.inspection import permutation_importance

# Note: This will take a bit of time
print("\nRunning permutation importance... (could be slow)")

importance_result = permutation_importance(
    model_pipeline,
    test_data,
    test_target,
    n_repeats=10,
    random_state=42,
    scoring="neg_root_mean_squared_error"
)

# Match importance to the raw column names
importance_scores = pd.Series(
    importance_result.importances_mean,
    index=test_data.columns
)

top_ranked_features = importance_scores.sort_values(ascending=False).head(15)

print("\nTop 15 Most Influential Features:")
print(top_ranked_features)

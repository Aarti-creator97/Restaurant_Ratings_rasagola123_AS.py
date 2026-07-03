# 🍽️ Restaurant Rating Prediction Using Machine Learning

This project develops a machine learning model to estimate restaurant ratings from structured restaurant data. The workflow includes data preparation, feature preprocessing, model training, performance evaluation, and identification of the most significant features influencing predictions.

## 🎯 Project Objective
The objective of this project is to predict the **Aggregate Rating** of restaurants by applying a **Gradient Boosting Regression** algorithm.

---

## ✨ Key Features

- Predicts restaurant ratings using supervised machine learning.
- Handles missing values automatically before model training.
- Standardizes numerical attributes and encodes categorical variables.
- Evaluates model performance using RMSE and R² metrics.
- Identifies the most important input features through permutation importance analysis.
- ## 📂 Dataset Information

- **Dataset File:** `Dataset.csv`
- **Target Variable:** `Aggregate rating`
- **Excluded Columns:**
  - `Rating color`
  - `Rating text`

These columns are removed because they provide direct information about the target value.

---

## 📚 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- ## ⚙️ Workflow

### 1. Data Import
The dataset is loaded from a CSV file and divided into predictor variables and the target variable.

### 2. Train-Test Split
The data is separated into training and testing subsets using an 80:20 ratio.

### 3. Data Preprocessing
The preprocessing pipeline performs:

- Median imputation for missing numerical values
- Standardization of numerical features
- Most frequent value imputation for categorical features
- One-Hot Encoding for categorical variables

### 4. Model Development
A **GradientBoostingRegressor** is trained using the processed training data.

### 5. Model Evaluation
The trained model is evaluated using:

- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R² Score)

### 6. Feature Importance
Permutation Importance is applied to determine the top 15 features that have the greatest impact on prediction performance.

---

## 📈 Example Output

```text
Model Performance

RMSE : 0.235
R² Score : 0.912

Top 15 Important Features
-------------------------
Votes
Average Cost for Two
Price Range
...
```

---

## ▶️ How to Run

1. Download or clone this repository.
2. Update the dataset path inside the Python script if necessary.
3. Install the required libraries.

```bash
pip install pandas numpy scikit-learn
```

4. Execute the program.

```bash
python Restaurant_Ratings.py
```

---

## 📌 Project Outcome

_This project demonstrates an end-to-end machine learning pipeline for restaurant rating prediction using Gradient Boosting Regression. 
It also provides insights into the most influential features affecting restaurant ratings, making the model both predictive and interpretable._

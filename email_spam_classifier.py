#Name: Muhammad Hamza Ashfaq
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Step 1: Loading the dataset
data = pd.read_csv("emails .csv")

# Previewing the data
print("First few rows of the dataset:")
print(data.head())

# Step 2: Data Cleaning and Preparation
# Droping non-feature columns if any, such as 'Email No. is the non-feature column meaning that it does not affect the result of target column!'
if 'Email No.' in data.columns:
    data = data.drop(columns=['Email No.'])

# Checking for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Handling Missing Values - Droping or Imputing missing values
data = data.dropna()

# Encoding categorical variables if present
# Assuming 'Prediction' is the target column, and it contains 0 (not spam) and 1 (spam)
label_encoder = LabelEncoder()
if data['Prediction'].dtype == 'object':
    data['Prediction'] = label_encoder.fit_transform(data['Prediction'])

# Separating features and target variable
X = data.drop(columns=['Prediction']) #Separating Prediction Column from other columns
y = data['Prediction']  # Now Assigning Prediction column to y

# Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Model Selection and Cross-Validation
# Define models to explore
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}  #Dictionary Of Models

# Evaluate models using cross-validation
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
    print(f"{model_name} - F1 Score (Cross-Validation): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Step 4: Model Training on Chosen Algorithm (choosing the best-performing model from the previous step)
""" Results are:
 Logistic Regression - F1 Score (Cross-Validation): 0.92 ± 0.02
 Decision Tree - F1 Score (Cross-Validation): 0.86 ± 0.03
 Support Vector Machine - F1 Score (Cross-Validation): 0.85 ± 0.02
 Random Forest - F1 Score (Cross-Validation): 0.93 ± 0.02
 Gradient Boosting - F1 Score (Cross-Validation): 0.92 ± 0.01 """
# I am Using Random Forest based on cross-validation results
# Highest F1 Score: Random Forest achieved an F1 score of 0.93, which is the highest among the algorithms tested, suggesting it performs well at balancing precision and recall.
# Low Variability: With a standard deviation of ±0.02, Random Forest shows stable performance across cross-validation folds, indicating reliability.
# However, Logistic Regression and Gradient Boosting are close contenders, both scoring 0.92, but with slightly higher or equal standard deviatio

model = RandomForestClassifier()

# Spliting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Predicting on test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance on Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# Step 6: Hyperparameter Tuning with Grid Search
# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize Grid Search with Cross-Validation
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from grid search
print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

# Evaluate the best model from grid search
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Calculate evaluation metrics for the tuned model
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print("\nTuned Model Performance on Test Set:")
print(f"Accuracy: {accuracy_best:.2f}")
print(f"Precision: {precision_best:.2f}")
print(f"Recall: {recall_best:.2f}")
print(f"F1 Score: {f1_best:.2f}")
print("\nClassification Report for Tuned Model:")
print(classification_report(y_test, y_pred_best, target_names=['Not Spam', 'Spam']))

# Install libraries if not already installed: python -m pip install pandas scikit-learn streamlit
# Importing necessary libraries
import pandas as pd

# Import dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Define bins and labels for price categories
bins = [0, 15, 25, df["medv"].max()]  # .max() to get the maximum value of the 'medv' column
labels = ['cheap', 'medium', 'expensive'] # 3 categories for the price

# Create price category column 
# df["price_category"] = pd.cut(df["medv"], bins=bins, labels=labels) #pd.cut() to create the price_category column based on the defined bins and labels

# Use pd.qcut() to create price categories based on quantiles, which will ensure that we have an equal number of instances in each category
# Model was weak to classify "expensive" if we used pd.cut(), so we will use pd.qcut() to create price categories based on quantiles, which will ensure that we have an equal number of instances in each category
# df["price_category"] = pd.qcut(df["medv"],q=3,labels=["cheap", "medium", "expensive"])

# Moving back to pd.cut(), in order to try out class_weight parameter in Random Forest Classifier to handle class imbalance
df["price_category"] = pd.cut(df["medv"], bins=bins, labels=labels)

'''
# Print the value counts and check for missing values in the price_category column
print(df["price_category"].value_counts()) # .value_counts() to see how many instances of each price category we have
print(df["price_category"].isna().sum()) # .isna().sum() to check for any missing values in the price_category column


Results:
price_category
medium       285
expensive    124
cheap         97
0 # -> No missing values in the price_category column

Next: define the features (X) and the target variable (y)
axis=1 to drop columns 'medv' and 'price_category' from the features, axis=0 would drop rows
Set the target variable to the price category
'''
features_to_keep = ["lstat", "rm", "crim", "nox", "indus"] # Selecting only the most important features based on feature importance from previous runs
X = df[features_to_keep]
y = df["price_category"] 

# Import necessary libraries for data preprocessing and modeling
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
# stratify=y to ensure that the distribution of price categories is maintained in both training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# Import the Random Forest Decision Tree Classifier model from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier model with specific hyperparameters
model = RandomForestClassifier(
    n_estimators=200,       #number of trees in the forest
    max_depth=15,           #maximum depth of the tree
    min_samples_split=15,   #minimum number of samples required to split an internal node
    min_samples_leaf=5,     #minimum number of samples required to be at a leaf node
    random_state=42,        #random seed for reproducibility
    class_weight="balanced" #assigning weights to classes to handle class imbalance
)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the price categories for the test set
y_pred = model.predict(X_test)

# Feature Importance - to understand which features are most important for the model's predictions
importances = model.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
print(feature_importance_df)

# Evaluate the model's performance using accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

'''
Results: 
1st Time Accuracy: 0.8921568627450981 (cut)
2nd Time Accuracy: 0.7745098039215687 (qcut)
3rd Time Accuracy: 0.9019607843137255 (cut with class_weight="balanced")
'''

# Calculate and print the confusion matrix to evaluate the model's performance in more detail
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

'''
Results Confusion Matrix:

1st Time (cut)  2nd Time (qcut)    3rd Time (cut with class_weight="balanced")
[[17  0  3]     [[25 2 7]          [[18  0  2]
 [ 1 18  6]     [ 0 27 6]          [ 0 21  4]
 [ 1  0 56]]    [ 3 5 27]]         [ 3  1 53]]
 Observation: The model performed better in classifying "expensive" category when we used pd.qcut() to create price categories based on quantiles
              Best results were achieved when we used pd.cut() with class_weight="balanced" in the Random Forest Classifier
 '''

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Grid Search for Hyperparameter Tuning - to find the best hyperparameters for the Random Forest Classifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Grid Search
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Initialize Grid Search with the Random Forest Classifier and the defined parameter grid
grid_search = GridSearchCV(
    estimator=model,        # using the previously defined Random Forest Classifier with class_weight="balanced"
    param_grid=param_grid,  # the parameter grid to search over
    cv=5,                   # 5-fold cross validation
    scoring="f1_macro",     # using F1 macro score as the scoring metric to evaluate the performance of the model with different hyperparameters
    n_jobs=-1               # using all available CPU cores to speed up the Grid Search process
)

# Fit Grid Search to the training data to find the best hyperparameters
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# Results: Best parameters: {'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
# As a result the accuracy and recall for "expensive" category dropped, so we will stick to the previous hyperparameters for the final model

'''
Results Classification Report:

1st Time (cut)
              precision    recall  f1-score   support

       cheap       0.89      0.85      0.87        20
   expensive       1.00      0.72      0.84        25
      medium       0.86      0.98      0.92        57

    accuracy                           0.89       102
   macro avg       0.92      0.85      0.88       102
weighted avg       0.90      0.89      0.89       102

2nd Time (qcut)
              precision    recall  f1-score   support

       cheap       0.89      0.74      0.81        34
   expensive       0.79      0.82      0.81        33
      medium       0.68      0.77      0.72        35

    accuracy                           0.77       102
   macro avg       0.79      0.77      0.78       102
weighted avg       0.79      0.77      0.78       102

3rd Time (cut with class_weight="balanced")
              precision    recall  f1-score   support

       cheap       0.86      0.90      0.88        20
   expensive       0.95      0.84      0.89        25
      medium       0.90      0.93      0.91        57

    accuracy                           0.90       102
   macro avg       0.90      0.89      0.90       102
weighted avg       0.90      0.90      0.90       102

Observation: recall for "expensive" category improved significantly when we used pd.qcut() 
             the best results were achieved when we used pd.cut() with class_weight="balanced" in the Random Forest Classifier
'''
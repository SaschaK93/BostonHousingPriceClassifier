# Install libraries if not already installed: python -m pip install pandas scikit-learn streamlit
# Importing necessary libraries
import pandas as pd

# Import dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Define bins and labels for price categories
bins = [0, 15, 25, df["medv"].max()]  # .max() to get the maximum value of the 'medv' column
labels = ['cheap', 'medium', 'expensive'] # 3 categories for the price
df["price_category"] = pd.cut(df["medv"], bins=bins, labels=labels)

features_to_keep = ["lstat", "rm", "crim", "nox", "indus"] # Selecting only the most important features based on feature importance from previous runs
X = df[features_to_keep]
y = df["price_category"] 

# Import necessary libraries for data preprocessing and modeling
from sklearn.model_selection import train_test_split
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

# Import joblib to save the trained model to a file
import joblib

# Save the trained model to a file named "model.pkl" using joblib
joblib.dump(model, "model.pkl")
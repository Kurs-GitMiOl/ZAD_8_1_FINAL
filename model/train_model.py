# Imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from pathlib import Path
# from sklearn.decomposition import PCA # uncomment if you want to reduce dimensions

# Uncomment the following prints to see how the Iris data looks

iris = load_iris()

df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

df['species'] = [iris['target_names'][i] for i in iris['target']]

# Preview the first 5 rows
# print("\nPreview the first 5 rows")
# print(df.head(5))

# Basic statistics of the dataset
# print("\nBasic statistics of the datase")
# print(df.describe())


# Load Iris data (features and labels) as DataFrame
X, y = load_iris(return_X_y=True, as_frame=True)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Pipeline: feature scaling + SVM classifier
# ('pca', PCA(n_components=3)) # add PCA to reduce dimensions if needed
model = Pipeline([
    ('scaler', StandardScaler()),                   # Feature scaling
    ('clf', SVC(kernel='linear', probability=True)) # SVM classifier
])


# Train the pipeline on the training data
model.fit(X_train, y_train)

# Check the model accuracy
# Uncomment print if needed
accuracy = model.score(X_test, y_test)
# print(f"\nAccuracy: {accuracy:.3f}")

# Save the trained pipeline to a file
MODEL_PATH = Path(__file__).parent / "iris_model.joblib"
joblib.dump(model, MODEL_PATH)
# print(f"Model saved to: {MODEL_PATH}")

# print("Model saved to model directory model/iris_model.joblib")

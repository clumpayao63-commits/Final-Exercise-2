import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Download the red-wine dataset from the UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")
df.to_csv("wine_quality.csv", index=False)

# 2. Inspect the dataset
print("Rows and columns:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nQuality distribution:")
print(df["quality"].value_counts().sort_index())

# 3. Remove duplicate rows and create the binary target
df = df.drop_duplicates().copy()
y = (df["quality"] >= 6).astype(int)  # 1 = Good; 0 = Needs improvement

# IMPORTANT: Do not use the original quality column as an input feature.
X = df.drop(columns=["quality"])

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Build preprocessing and Random Forest model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

# 6. Train and evaluate the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Needs improvement", "Good quality"],
    zero_division=0
))

# 7. Save the trained model
joblib.dump(model, "wine_model.pkl")
print("\nSaved wine_model.pkl")
print("Dataset saved as wine_quality.csv")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data/parkinsons.csv")
X = data.drop(columns=["name", "status"])
y = data["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/rf_model.pkl")

# Generate feature importance plot
plt.figure(figsize=(10,6))
pd.Series(model.feature_importances_, index=X.columns)\
  .nlargest(10)\
  .plot(kind='barh')
plt.title("Top 10 Important Features")
plt.savefig("models/feature_importance.png", bbox_inches='tight')
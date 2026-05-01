# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("merged ai and job market.csv")


# 2. ENCODE EXPERIENCE
# Simple manual encoding
experience_map = {
    "entry": 0,
    "mid": 1,
    "senior": 2
}

df["experience_encoded"] = df["experience"].map(experience_map)

# Drop rows that didn't map correctly
df = df.dropna(subset=["experience_encoded"])


# 3. DEFINE FEATURES + TARGET
X = df[["experience_encoded"]]
y = df["salary_usd"]


# 4. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)


# 6. MAKE PREDICTIONS
y_pred = model.predict(X_test)


# 7. EVALUATION
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)


# 8. INTERPRET MODEL
print("Intercept:", model.intercept_)
print("Coefficient (experience):", model.coef_[0])


# Imports
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Average scores
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

# Plot
plt.figure()
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Validation Score")

plt.xlabel("Training Size", fontweight='bold')
plt.ylabel("R² Score", fontweight='bold')
plt.title("Learning Curve", fontweight='bold')

plt.legend()
plt.show()


# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# loading the dataset
df = pd.read_csv("merged ai and job market.csv")


# simple encoding of experience
experience_map = {
    "entry": 0,
    "mid": 1,
    "senior": 2
}

df["experience_encoded"] = df["experience"].map(experience_map)

# delete rows that don't map correctly
df = df.dropna(subset=["experience_encoded"])


# defining features and target
X = df[["experience_encoded"]]
y = df["salary_usd"]


# splitting the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# training the model
model = LinearRegression()
model.fit(X_train, y_train)


# predictions
y_pred = model.predict(X_test)


# evaluating the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)


# interpreting the model
print("Intercept:", model.intercept_)
print("Coefficient (experience):", model.coef_[0])


# imports
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# creating a learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10))


# average scores
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)


# plot
plt.figure()
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Validation Score")

plt.xlabel("Training Size", fontweight='bold')
plt.ylabel("R² Score", fontweight='bold')
plt.title("Learning Curve", fontweight='bold')

plt.legend()
plt.show()


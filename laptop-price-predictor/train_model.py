import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("cleaned_laptop_data.csv")

X = df.drop(columns=["Price", "high_price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

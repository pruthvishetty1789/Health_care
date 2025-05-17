import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Diabetes_prediction.csv")


X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)


with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Diabetes model and scaler saved successfully.")

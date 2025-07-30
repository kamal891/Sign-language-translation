import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 📌 Load webcam-only data
df = pd.read_csv('data/sign_data.csv')

# Drop duplicate header rows if they exist
df = df[df['label'] != 'label']

X = df.drop('label', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy * 100:.2f}%")

joblib.dump(clf, 'sign_model.pkl')
print("✅ Model saved as 'sign_model.pkl'")

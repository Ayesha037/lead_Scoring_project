import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Lead Intelligence Dashboard", layout="wide")

st.title("🚀 Lead Intelligence & Conversion Prediction")


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "facebook_ads.csv")


if not os.path.exists(csv_path):
    st.error(f"❌ CSV file not found!")
    st.info(f"Looking for: {csv_path}")
    st.warning("Make sure 'facebook_ads.csv' is in the same folder as this script and committed to GitHub")
    st.stop()

df = pd.read_csv(csv_path)

df.columns = df.columns.str.lower().str.strip()

df.rename(columns={
    "time spent on site": "time_spent"
}, inplace=True)

required = ["time_spent", "salary", "clicked"]

missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

col1.metric("Total Users", len(df))
col2.metric("Conversion Rate", f"{df['clicked'].mean()*100:.2f}%")

st.dataframe(df.head())

st.subheader("📊 Behavioral Insights")

clicked = df[df["clicked"] == 1]
not_clicked = df[df["clicked"] == 0]

col1, col2 = st.columns(2)

col1.write("### ⏱ Time Spent")
col1.write(f"Clicked: {clicked['time_spent'].mean():.2f}")
col1.write(f"Not Clicked: {not_clicked['time_spent'].mean():.2f}")

col2.write("### 💰 Salary")
col2.write(f"Clicked: {clicked['salary'].mean():.2f}")
col2.write(f"Not Clicked: {not_clicked['salary'].mean():.2f}")

st.subheader("📈 Data Visualization")

fig1, ax1 = plt.subplots()
ax1.hist([clicked["time_spent"], not_clicked["time_spent"]], label=["Clicked", "Not Clicked"])
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.hist([clicked["salary"], not_clicked["salary"]], label=["Clicked", "Not Clicked"])
ax2.legend()
st.pyplot(fig2)

st.subheader("🤖 Machine Learning Model")

X = df[["time_spent", "salary"]]
y = df["clicked"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {accuracy*100:.2f}%")

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig3, ax3 = plt.subplots()
ax3.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm)):
        ax3.text(j, i, cm[i][j], ha="center", va="center")

ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

st.pyplot(fig3)

st.subheader("🎯 Predict User Click")

time_input = st.slider("Time Spent", 0, int(df["time_spent"].max()))
salary_input = st.slider("Salary", 0, int(df["salary"].max()))

prediction = model.predict([[time_input, salary_input]])[0]
prob = model.predict_proba([[time_input, salary_input]])[0][1]

if prediction == 1:
    st.success(f"🔥 Likely to CLICK ({prob*100:.2f}%)")
else:
    st.warning(f"⚠️ Not likely to click ({prob*100:.2f}%)")

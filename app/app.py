import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load and prepare dataset
# -------------------------------
df = pd.read_csv("https://github.com/AstosM/Diabetes-Prediction/blob/main/app/diabetes.csv")

X = df.drop(columns="Outcome", axis=1)
Y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Accuracy scores
train_acc = accuracy_score(Y_train, classifier.predict(X_train))
test_acc = accuracy_score(Y_test, classifier.predict(X_test))

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

st.title("🩺 Diabetes Prediction App")
st.markdown("A simple ML app using **Support Vector Machine (SVM)** to predict diabetes.")

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Dataset & Analysis", "🤖 Prediction"])

# -------------------------------
# Home Section
# -------------------------------
if section == "🏠 Home":
    st.subheader("About this project")
    st.write("""
    This app uses the **PIMA Diabetes Dataset** and trains a Support Vector Machine model
    to predict whether a person is diabetic or not based on health measurements.
    """)

    st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
    st.metric("Testing Accuracy", f"{test_acc*100:.2f}%")

# -------------------------------
# Dataset & Analysis Section
# -------------------------------
elif section == "📊 Dataset & Analysis":
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.write("### Distribution of Outcome (0 = Non-Diabetic, 1 = Diabetic)")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Outcome", ax=ax, palette="Set2")
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------
# Prediction Section
# -------------------------------
elif section == "🤖 Prediction":
    st.subheader("Enter Patient Details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose", min_value=0, max_value=300, value=103)
            bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=30)
            skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=38)

        with col2:
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=83)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=43.3, format="%.1f")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.183, format="%.3f")
            age = st.number_input("Age", min_value=1, max_value=120, value=33)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        std_data = scaler.transform(input_data)
        prediction = classifier.predict(std_data)

        if prediction[0] == 0:
            st.success("✅ The person is **Not Diabetic**")
        else:
            st.error("⚠️ The person is **Diabetic**")

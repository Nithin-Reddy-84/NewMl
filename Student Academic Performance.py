import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Predict student performance and understand influencing factors")

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

# SHAP initialization
shap.initjs()

# ==============================
# USER INPUTS
# ==============================
st.subheader("📥 Enter Student Details")

gender = st.selectbox("Gender", ["Male", "Female"])
study_hours = st.slider("Study Hours per Week", 0, 20, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_grade = st.slider("Previous Grade", 0, 100, 60)
activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
parent_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])

# ==============================
# ENCODING INPUTS
# ==============================
gender = 1 if gender == "Male" else 0
activities = 1 if activities == "Yes" else 0

parent_map = {"Low": 0, "Medium": 1, "High": 2}
parent_support = parent_map[parent_support]

input_data = pd.DataFrame([[
    gender,
    attendance,
    study_hours,
    previous_grade,
    activities,
    parent_support
]], columns=[
    "Gender",
    "Attendance",
    "StudyHours",
    "PreviousGrade",
    "ExtracurricularActivities",
    "ParentalSupport"
])

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict Performance"):

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    result = label_map[prediction]

    st.subheader("📊 Prediction Result")
    st.success(f"Performance Level: {result}")

    st.write("### 🔢 Prediction Probability")
    st.write({
        "Low": round(probability[0][0], 2),
        "Medium": round(probability[0][1], 2),
        "High": round(probability[0][2], 2)
    })

    # ==============================
    # SHAP EXPLANATION
    # ==============================
    st.subheader("🧠 Model Explanation (SHAP)")

    explainer = shap.Explainer(model, scaler.transform)
    shap_values = explainer(scaled_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

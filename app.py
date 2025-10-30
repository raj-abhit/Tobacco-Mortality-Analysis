import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# 🎯 App Configuration
# ==============================================
st.set_page_config(page_title="Tobacco Use & Mortality Prediction",
                   page_icon="🚭",
                   layout="wide")

# ==============================================
# 🎨 Header
# ==============================================
st.title("🚭 Tobacco Use & Mortality Prediction App")
st.markdown("""
This web application helps analyze and predict **tobacco-related health outcomes**
using Machine Learning techniques.

Developed by **Abhit Raj**.
""")

# ==============================================
# 📦 Load Data and Model
# ==============================================
@st.cache_data
def load_data():
    df = pd.read_csv("merged_cleaned.csv")
    # Convert possible numeric-like strings to numbers
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Model could not be loaded: {e}")
        return None

df = load_data()
model = load_model()

# ==============================================
# 📊 Sidebar Navigation
# ==============================================
tabs = st.tabs(["📈 EDA Preview", "🤖 Model Prediction", "ℹ️ About Project"])

# ==============================================
# 📊 TAB 1: Exploratory Data Analysis
# ==============================================
with tabs[0]:
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.write("Here’s a quick overview of your dataset:")

    st.dataframe(df.head())
    st.write("**Shape:**", df.shape)

    st.subheader("🔹 Missing Values")
    st.write(df.isnull().sum())

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        st.subheader("📉 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), cmap="coolwarm", annot=False)
        st.pyplot(fig)

        st.subheader("📊 Distribution of Key Numeric Columns")
        sel_col = st.selectbox("Select a column to view distribution:", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[sel_col], bins=30, kde=True)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric columns found for EDA visualization.")

# ==============================================
# 🤖 TAB 2: Model Prediction
# ==============================================
with tabs[1]:
    st.header("🤖 Predict Using Trained ML Model")

    if model is None:
        st.warning("⚠️ Model not found. Please ensure 'best_model.pkl' exists.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            st.warning("⚠️ Not enough numeric columns for prediction.")
        else:
            st.write("Enter feature values below:")

            # Split input columns into two halves for cleaner layout
            col1, col2 = st.columns(2)
            inputs = {}
            for i, col in enumerate(num_cols):
                if i % 2 == 0:
                    inputs[col] = col1.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    inputs[col] = col2.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

            input_df = pd.DataFrame([inputs])

            if st.button("🚀 Predict"):
                try:
                    pred = model.predict(input_df)[0]
                    st.success(f"✅ Predicted Value: **{pred:.2f}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ==============================================
# ℹ️ TAB 3: About Project
# ==============================================
with tabs[2]:
    st.header("ℹ️ About This Project")
    st.markdown("""
    **Project Title:** Tobacco Use and Mortality (2004–2015)

    **Objective:**  
    Analyze tobacco use, prescriptions, and health metrics data to
    understand correlations and predict outcomes related to tobacco-related mortality.

    **Tech Stack:**  
    - Python 🐍  
    - Pandas, NumPy, Scikit-learn  
    - Matplotlib, Seaborn  
    - Streamlit for deployment

    **Developed by:** Abhit Raj  
    
    """)

# ==============================================
# ✨ Footer
# ==============================================
st.markdown("---")
st.markdown("© 2025 | Built with ❤️ by Abhit Raj | Powered by Streamlit")

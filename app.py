import streamlit as st
import joblib
import json
import numpy as np

st.set_page_config(page_title="Tobacco Use & Mortality Prediction", page_icon="🚭", layout="wide")

st.markdown(
    "<h1 style='text-align:center; color:#FF4B4B;'>🚭 Tobacco Use & Mortality Prediction</h1>",
    unsafe_allow_html=True,
)

try:
    model = joblib.load(open("best_model.pkl", "rb"))
    with open("feature_order.json", "r", encoding="utf-8") as f:
        features = json.load(f)
    st.success("✅ Model and feature order loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or feature file:\n\n{e}")
    st.stop()

st.sidebar.header("📊 Input Parameters")

inputs = []
for feature in features:
    value = st.sidebar.number_input(feature, value=0.0)
    inputs.append(value)

if st.button("🧮 Predict Mortality Rate"):
    try:
        X_input = np.array(inputs).reshape(1, -1)
        prediction = model.predict(X_input)
        st.success(f"### ✅ Predicted Mortality Rate: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

st.markdown("<br><center>Built with ❤️ by Abhit Raj | Powered by Streamlit & Scikit-learn</center>", unsafe_allow_html=True)

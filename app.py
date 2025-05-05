import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("parkinson_model.pkl")
scaler = joblib.load("parkinson_scaler.pkl")

# UI Config
st.set_page_config(page_title="Parkinson's Detector", layout="wide")

# --- Header ---
st.title("🧠 Parkinson's Disease Detection")
st.markdown("""
This app uses machine learning to detect Parkinson's Disease based on voice parameters.  
Enter the details below or upload a CSV file to get predictions.
""")

# --- Input Section ---
st.subheader("🎙️ Enter Voice Parameters:")

# Create horizontal columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    fo = st.number_input("MDVP:Fo (Hz)", value=119.992)
    ddp = st.number_input("Jitter:DDP", value=0.009)
    apq5 = st.number_input("Shimmer:APQ5", value=0.03)
    hnr = st.number_input("HNR", value=21.033)
    spread2 = st.number_input("spread2", value=0.22)

with col2:
    fhi = st.number_input("MDVP:Fhi (Hz)", value=157.302)
    shimmer = st.number_input("MDVP:Shimmer", value=0.02)
    dda = st.number_input("Shimmer:DDA", value=0.02)
    rpde = st.number_input("RPDE", value=0.5)
    d2 = st.number_input("D2", value=2.3)

with col3:
    flo = st.number_input("MDVP:Flo (Hz)", value=74.997)
    shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.2)
    apq = st.number_input("MDVP:APQ", value=0.04)
    dfa = st.number_input("DFA", value=0.7)
    ppe = st.number_input("PPE", value=0.3)

with col4:
    jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.005)
    apq3 = st.number_input("Shimmer:APQ3", value=0.04)
    nhr = st.number_input("NHR", value=0.023)
    spread1 = st.number_input("spread1", value=-5.0)
    jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.0001)

with col5:
    rap = st.number_input("MDVP:RAP", value=0.003)
    ppq = st.number_input("MDVP:PPQ", value=0.02)

# --- Prediction Button ---
if st.button("Predict"):
    input_data = np.array([[
        fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
        shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
        rpde, dfa, spread1, spread2, d2, ppe
    ]])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("🔍 Result:")
    if pred == 1:
        st.error(f"⚠️ Parkinson's Detected (confidence: {proba:.1%})")
    else:
        st.success(f"✅ No Parkinson's Detected (confidence: {1 - proba:.1%})")

# --- Bulk Prediction ---
st.markdown("---")
st.subheader("📂 Bulk Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with 22 columns (no 'name')", type="csv")

expected_cols = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
                 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
                 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
                 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2',
                 'D2', 'PPE']

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if all(col in df.columns for col in expected_cols):
            df_scaled = scaler.transform(df[expected_cols])
            df['Prediction'] = model.predict(df_scaled)
            df['Confidence'] = model.predict_proba(df_scaled)[:, 1]

            st.dataframe(df)
            st.download_button("Download Results", df.to_csv(index=False), "parkinsons_predictions.csv")
        else:
            st.error("Uploaded CSV is missing one or more required columns.")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

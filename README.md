# 🧠 Parkinson's Disease Detection App

A machine learning-powered web app built with **Streamlit** to detect Parkinson's Disease based on voice measurements.

## 📌 Overview

This project uses biomedical voice measurements to determine whether a person has Parkinson's Disease. The app allows users to:
- Manually input voice features and get a prediction
- Upload a CSV file for bulk predictions
- View clear results with an intuitive UI

## 💻 Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Joblib

## 📂 Dataset

The dataset used is publicly available on [Kaggle](https://www.kaggle.com/datasets). It contains various voice features collected from healthy individuals and people with Parkinson’s Disease.

## 🧪 Features Used for Prediction

- MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
- Jitter(%) and Jitter(Abs)
- Shimmer, Shimmer(dB), RAP, PPQ, DDP, DDA
- NHR, HNR
- RPDE, DFA, Spread1, Spread2, D2
- APQ3, APQ5, APQ

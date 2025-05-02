import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load models
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca_model.pkl", "rb"))
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))

# Page setup
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")
st.title("🛍️ Mall Customer Segmentation")
st.markdown("""
Predicts which customer group a person falls into using **KMeans Clustering**  
based on **Gender, Age, Annual Income, and Spending Score**.
""")

# Sidebar input
st.sidebar.header("🧾 Customer Info")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender_value = 1 if gender == "Male" else 0

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 60)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Prediction
if st.sidebar.button("🔍 Segment Customer"):
    columns = ['Gender', 'Age', 'Annual Income', 'Spending Score']
    input_df = pd.DataFrame([[gender_value, age, income, score]], columns=columns)
    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"🎯 This customer belongs to **Cluster {cluster}**")

    # Optional Visualization
    st.markdown("### 🧠 Cluster Projection using PCA")

    sample_data = pd.DataFrame({
        'Gender': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
        'Age': [19, 21, 35, 40, 23, 31, 50, 22, 65, 29],
        'Annual Income': [15, 80, 35, 130, 45, 65, 90, 100, 120, 55],
        'Spending Score': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
    })

    sample_scaled = scaler.transform(sample_data)
    sample_pca = pca.transform(sample_scaled)
    sample_clusters = kmeans.predict(sample_scaled)

    input_pca = pca.transform(input_scaled)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample_pca[:, 0], sample_pca[:, 1], c=sample_clusters, cmap='tab10', alpha=0.6)
    ax.scatter(input_pca[0, 0], input_pca[0, 1], color='red', s=200, edgecolors='black', label='New Customer')
    ax.set_title("Customer Clusters (PCA Projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    st.pyplot(fig)

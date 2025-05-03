import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- SETUP --------------------
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Customer Segmentation with Clustering")
st.markdown("Predict customer clusters using **KMeans** and visualize them in **PCA space**.")

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    pca = pickle.load(open("pca_model.pkl", "rb"))
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    return scaler, pca, kmeans

scaler, pca_model, kmeans_model = load_models()

# -------------------- USER INPUT --------------------
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 15, 70, 30)
income = st.sidebar.slider("Annual Income", 15, 150, 60)
score = st.sidebar.slider("Spending Score", 1, 100, 50)

if st.sidebar.button("Predict Cluster"):
    # -------------------- PREDICT --------------------
    gender_encoded = 1 if gender == "Male" else 0
    user_input = np.array([[gender_encoded, age, income, score]])
    scaled_input = scaler.transform(user_input)
    cluster = kmeans_model.predict(scaled_input)[0]
    pca_input = pca_model.transform(scaled_input)

    st.subheader("🎯 Prediction Result")
    st.write(f"The customer is predicted to belong to **Cluster {cluster}**.")

    # -------------------- CLUSTER VISUALIZATION --------------------
    st.subheader("📊 PCA Cluster Visualization")

    # Generate data for plotting
    # Load sample data used during training if available
    try:
        df = pd.read_csv("Mall_Customers.csv")
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        features = df[['Gender', 'Age', 'Annual Income', 'Spending Score']]
        features.columns = ['Gender', 'Age', 'Annual Income', 'Spending Score']
        X_scaled = scaler.transform(features)
        X_pca = pca_model.transform(X_scaled)
        clusters = kmeans_model.predict(X_scaled)

        plot_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        plot_df["Cluster"] = clusters

        # Add user input to the plot
        user_df = pd.DataFrame(pca_input, columns=["PC1", "PC2"])
        user_df["Cluster"] = ["User Input"]

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster", palette="Set1", alpha=0.6)
        plt.scatter(user_df["PC1"], user_df["PC2"], c='black', s=150, marker='X', label='User')
        plt.legend()
        plt.title("PCA Clusters with User Input")
        st.pyplot(plt)

    except FileNotFoundError:
        st.warning("Training data not found. Please ensure 'Mall_Customers.csv' is available.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
import plotly.express as px
from joblib import dump

cl = {0: "Class A", 1: "Class B", 2: "Class C"}

msg = "Saved Successfully In 'model' folder "
model = KMeans(n_clusters=3, random_state=42)


def encode(df):
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    df_encoder = pd.DataFrame(
        encoder.fit_transform(df[c_cols]), columns=encoder.get_feature_names_out(c_cols)
    )
    return df_encoder


def scale(df):
    df_scaler = StandardScaler().fit_transform(df[n_cols])
    df_scaler = pd.DataFrame(df_scaler, columns=n_cols)
    return df_scaler


def config():
    st.set_page_config(
        layout="wide",
        page_title="Customer Segmentation App",
        page_icon="📊",
        initial_sidebar_state="expanded",
    )
    st.title("Customer Segmentation (KMeans Clustering) 📊")
    st.sidebar.header("Configuration")
    st.sidebar.markdown(
        "Upload your dataset and select numerical and categorical columns for clustering."
    )


config()

uploaded_file = st.file_uploader("Upload your file ", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)

    n_cols = st.sidebar.multiselect(
        "Choose numerical columns:",
        options=df.select_dtypes(include=["int64", "float64"]).columns.tolist(),
    )

    c_cols = st.sidebar.multiselect(
        "Choose categorical columns:",
        options=df.select_dtypes(exclude=["int64", "float64"]).columns.tolist(),
    )

    if n_cols and c_cols:
        try:
            en = encode(df)
            sc = scale(df)
            df_final = pd.concat((sc, en), axis=1)
            df["cluster"] = model.fit_predict(df_final)
            df2 = df.copy()
            df["cluster"] = df["cluster"].map(cl)

            st.dataframe(df)
            st.dataframe(df.groupby("cluster")[n_cols].mean())

            fig = px.imshow(
                df2[["cluster"] + n_cols].corr(),
                text_auto=".3f",
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Error processing the data:" + str(e))
            st.stop()

# Dump Model
if st.sidebar.button("Save KMeans Model"):
    dump(model, r"model\Kmeans_model.pkl")
    st.sidebar.success(msg)

# Save DataFrame
if st.sidebar.button("Save DataFrame as Excel"):
    df.to_excel(r"model\clustered_data.xlsx", index=False)
    st.sidebar.success(msg)

# Save Heatmap Plot
if st.sidebar.button("Save Heatmap Plot"):
    plt.savefig(r"model\heatmap.png", dpi=300, bbox_inches="tight")
    st.sidebar.success(msg)

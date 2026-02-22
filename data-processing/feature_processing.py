"""
Offline script: trains PCA on heart_features.csv and saves models to ../models/.

Run once to regenerate models:
    python feature_processing.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from condition_analysis import FEATURE_ORDER


def run_pca(df: pd.DataFrame):
    X = df[FEATURE_ORDER].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = df.copy()
    df_pca["PC1"] = X_pca[:, 0]
    df_pca["PC2"] = X_pca[:, 1]

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    return df_pca, pca, scaler


if __name__ == "__main__":
    df = pd.read_csv("heart_features.csv")
    df_pca, pca_model, scaler_model = run_pca(df)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="Category",
        palette="Set1",
        s=100,
    )
    plt.title("Morphology PCA Map")
    plt.show()

    joblib.dump(pca_model, "../models/pca_model.pkl")
    joblib.dump(scaler_model, "../models/scaler_model.pkl")
    df_pca.to_csv("../models/pca_dataset.csv", index=False)
    print("Models saved to ../models/")

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart_features.csv")


def analyze_severity_signal(df):
    
    # Select numeric feature columns only
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Remove identifiers
    feature_cols = [col for col in feature_cols if col not in ["Patient", "Pat", "Age"]]
    
    results = []
    
    categories = df["Category"].unique()
    
    for feature in feature_cols:
        
        groups = []
        for cat in categories:
            groups.append(df[df["Category"] == cat][feature])
        
        #ANOVA test
        try:
            f_stat, p_value = f_oneway(*groups)
        except:
            continue
        
        results.append({
            "Feature": feature,
            "F_statistic": f_stat,
            "p_value": p_value
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F_statistic", ascending=False)
    
    return results_df

def run_pca(df):

    feature_cols = [
        "Label_1_vol_ml",   # LV
        "Label_2_vol_ml",   # RV
        "Label_4_vol_ml",   # RA
        "Label_6_vol_ml",   # PA
        "Total_heart_vol"
    ]

    X = df[feature_cols].values

    # Standardizing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = df.copy()
    df_pca["PC1"] = X_pca[:, 0]
    df_pca["PC2"] = X_pca[:, 1]

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    return df_pca, pca, scaler

#signal_df = analyze_severity_signal(df)
df_pca, pca_model, scaler_model = run_pca(df)

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="Category",
    palette="Set1",
    s=100
)
plt.title("Morphology PCA Map")
plt.show()
# print(signal_df.head(10))
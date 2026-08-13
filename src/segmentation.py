import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_customer_clusters(
    feature_df: pd.DataFrame,
    n_clusters: int = 3,
    model_dir: str = "models",
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Scales load features, clusters customers into load profile archetypes using K-Means,

    and saves trained artifacts to disk.
    """
    df = feature_df.copy()

    X = df[cluster_cols]

    # Normalize features so large scales don't dominate distance metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    df["silhouette_score"] = round(float(sil_score), 4)

    # Save model artifacts for production reuse in app.py
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    print(f"Models saved to directory: '{model_dir}/'")

    return df, kmeans, scaler

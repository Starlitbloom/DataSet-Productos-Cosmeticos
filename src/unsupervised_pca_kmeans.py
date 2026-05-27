"""src.unsupervised_pca_kmeans

Análisis no supervisado para el dataset ya preprocesado:
- PCA para reducción de dimensionalidad
- K-Means para clustering
- Visualización 2D de clusters en espacio PCA
- Métrica Silhouette (coeficiente de Silhouette)

Importante:
Este script asume que ya existe la matriz numérica lista para modelado en:
  data/processed/sephora_limpio.csv

Esa matriz proviene del pipeline previo (escalado + OneHotEncoder).
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_matrix(csv_path: str) -> np.ndarray:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró el archivo: {csv_path}. "
            "Ejecuta antes el preprocessing para generar data/processed/sephora_limpio.csv"
        )

    df = pd.read_csv(csv_path)

    # Asegura conversión a numérico (por seguridad ante tipos raros)
    X = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    # Si hubiera NaNs por coerción (no esperado), reemplaza por 0 para evitar fallos.
    # (Idealmente el CSV ya viene sin NaNs, pero esto mantiene robustez.)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)

    return X


def run_pca_kmeans(
    X: np.ndarray,
    n_components: int,
    n_clusters: int,
    random_state: int,
    silhouette_sample_size: int | None = 5000,
) -> tuple[np.ndarray, np.ndarray, PCA, KMeans, float]:
    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_pca)

    # Silhouette
    if silhouette_sample_size is None:
        sil = silhouette_score(X_pca, cluster_labels)
    else:
        rs = np.random.RandomState(random_state)
        n = X_pca.shape[0]
        k = min(silhouette_sample_size, n)
        idx = rs.choice(n, size=k, replace=False)
        sil = silhouette_score(X_pca[idx], cluster_labels[idx])

    return X_pca, cluster_labels, pca, kmeans, float(sil)


def plot_clusters_pca2d(X_pca: np.ndarray, labels: np.ndarray, out_path: str | None = None) -> None:
    if X_pca.shape[1] < 2:
        raise ValueError("Se requieren al menos 2 componentes en PCA para graficar clusters en 2D.")

    df_plot = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "cluster": labels.astype(str),
        }
    )

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="cluster", palette="tab10", s=18, alpha=0.85)
    plt.title("Clusters visualizados en el espacio PCA (PC1 vs PC2)")
    plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=200)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA + KMeans sobre datos ya preprocesados.")
    parser.add_argument("--csv", default="data/processed/sephora_limpio.csv", help="Ruta al CSV preprocesado")
    parser.add_argument("--n-components", type=int, default=10, help="Componentes PCA")
    parser.add_argument("--n-clusters", type=int, default=6, help="Número de clusters KMeans")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5000,
        help="Muestra para silhouette_score (para acelerar). Usa 0 o None para calcular sobre todo.",
    )
    parser.add_argument(
        "--save-plot",
        default="data/processed/pca_kmeans_clusters.png",
        help="Ruta donde guardar el gráfico (PNG). Usa vacío ('') para no guardar.",
    )

    args = parser.parse_args()

    silhouette_sample_size = args.silhouette_sample_size
    if silhouette_sample_size in (None, 0):
        silhouette_sample_size = None

    X = load_matrix(args.csv)

    X_pca, labels, pca, kmeans, sil = run_pca_kmeans(
        X=X,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        silhouette_sample_size=silhouette_sample_size,
    )

    print("\n========== RESULTADOS PCA + KMEANS ==========")
    print(f"Matriz de entrada: {X.shape[0]} filas × {X.shape[1]} features")
    print(f"PCA: n_components={args.n_components}")
    print(f"Varianza explicada acumulada: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"KMeans: n_clusters={args.n_clusters}")
    print(f"Inertia (in-sample): {kmeans.inertia_:.4f}")
    print(f"Silhouette score: {sil:.4f}")
    print("==============================================\n")

    save_path = args.save_plot.strip() if isinstance(args.save_plot, str) else ""
    plot_clusters_pca2d(X_pca, labels, out_path=(save_path if save_path else None))


if __name__ == "__main__":
    main()


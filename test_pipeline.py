import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from gtda.plotting import plot_point_cloud
from persim import plot_diagrams

# === Import functions FROM YOUR REPO ===
from data import generate_coupled_pendulum, generate_uncoupled_pendulum
from src.tda import takens_embedding, compute_pd, compute_pl


def plot_time_series(theta, phi, title):
    t = np.arange(len(theta))
    plt.figure(figsize=(8, 3))
    plt.plot(t, theta, label=r"$\theta$")
    plt.plot(t, phi, label=r"$\phi$")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()
  
def main():

    # -----------------------------
    # 1. Generate datasets
    # -----------------------------
    combined_c, theta_c, phi_c = generate_coupled_pendulum(n_steps=2000)
    combined_u, theta_u, phi_u = generate_uncoupled_pendulum(n_steps=2000)

    plot_time_series(theta_c, phi_c, "Coupled Pendulum")
    plot_time_series(theta_u, phi_u, "Uncoupled Pendulum")

    # -----------------------------
    # 2. Takens embedding
    # -----------------------------
    X_c = takens_embedding(combined_c, dim=3, delay=10)
    X_u = takens_embedding(combined_u, dim=3, delay=10)

    plot_point_cloud(X_c, title="Takens Embedding (Coupled)")
    plot_point_cloud(X_u, title="Takens Embedding (Uncoupled)")

    # -----------------------------
    # 3. Persistence diagrams
    # -----------------------------
    dgms_c = compute_pd(X_c)
    dgms_u = compute_pd(X_u)

    plot_diagrams(dgms_c, title="PD (Coupled)")
    plot_diagrams(dgms_u, title="PD (Uncoupled)")

    # -----------------------------
    # 4. Persistence landscapes
    # -----------------------------

    # landscape functions are converted to vectors by simply discretizing
    v_c = compute_pl(dgms_c)
    v_u = compute_pl(dgms_u)

    # -----------------------------
    # 5. Build dataset
    # -----------------------------
    X = np.vstack([v_c, v_u])
    y = np.array([1] * len(v_c) + [0] * len(v_u))

    # -----------------------------
    # 6. PCA + SVM
    # -----------------------------

    test_on_svm(v_c, v_u)


if __name__ == "__main__":
    main()

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
from data.coupled_double_pendulum import generate_coupled_pendulum
from data.unucoupled_double_pendulum import generate_uncoupled_pendulum
from src.tda.takens_embedding import takens_embedding
from src.tda.compute_pd import compute_pd
from src.tda.compute_pl import compute_pl
from src.ml.SVM_PCA import test_on_svm


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
    N = 50
    combined_c, theta_c, phi_c = generate_coupled_pendulum(N, spacing=200)[0:3]
    combined_u, theta_u, phi_u = generate_uncoupled_pendulum(N, spacing=200)[0:3]

    # Plot example time series
    #index = 10
    #plot_time_series(theta_c[index], phi_c[index], "Coupled Pendulum")
    #print(theta_u[index])
    #print(phi_u[index])
    #plot_time_series(theta_u[index], phi_u[index], "Uncoupled Pendulum")

    # -----------------------------
    # 2. Takens embedding
    # -----------------------------
    X_c = takens_embedding(combined_c, N)
    X_u = takens_embedding(combined_u, N)

    X_c = np.array(X_c)
    X_u = np.array(X_u)

    #plot_point_cloud(X_c)#, title="Takens Embedding (Coupled)")
    #plot_point_cloud(X_u, title="Takens Embedding (Uncoupled)")

    # -----------------------------
    # 3. Persistence diagrams
    # -----------------------------
    dgms_c = compute_pd(X_c)
    dgms_u = compute_pd(X_u)

    #plot_diagram(dgms_c)
    #plot_diagram(dgms_u)

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

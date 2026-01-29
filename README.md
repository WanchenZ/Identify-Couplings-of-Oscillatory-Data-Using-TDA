# Identify Couplings of Oscillatory Data Using TDA

# 👤 About the Author

I am a Mathematics PhD candidate (graduating May 2026) specializing in the intersection of Algebraic Topology and Machine Learning. My work focuses on translating abstract mathematical structures into robust, scalable, and interpretable features for complex data pipelines. This repository demonstrates my ability to implement advanced mathematical theory into an end-to-end Python-based ML workflow.

_I am actively seeking roles in Data Science, Machine Learning Research, and Quantitative Analysis. Feel free to reach out via LinkedIn or Email._

# 🚀 Project Overview

This project addresses the challenge of identifying coupling behavior in high-dimensional oscillatory systems. Traditional signal processing often struggles with non-linear dependencies and noise. By leveraging Topological Data Analysis (TDA), we represent oscillatory signals as point clouds in a high-dimensional state space and use their "shape" to classify their coupling state.

Key Technical Achievements:

- **Dimensionality Reduction & State Space Reconstruction**: Implemented time-delay embeddings to transform 2D signals into high-dimensional manifolds.

- **Interactive 3D Visualization:** Developed a custom visualization pipeline using Plotly and PCA to project high-dimensional Takens' embeddings into an interactive 3D space, allowing for the intuitive inspection of attractor geometries.

- **Persistent Homology at Scale**: Developed optimized batch-processing functions using giotto-tda to compute Vietoris-Rips persistence diagrams across $H_0$, $H_1$, and $H_2$ dimensions.

- **Feature Vectorization:** Engineered a pipeline to transform abstract persistence landscapes into flattened feature vectors compatible with standard Scikit-Learn estimators.

- **Statistical Validation**: Implemented an SVM-based classification pipeline with k-fold cross-validation and 3D visualization of the decision hyperplane in PCA-reduced space.

# 🔬 Methodology & Pipeline

The core of this project is a robust pipeline that transforms raw dynamical system signals into high-dimensional topological features.

1. Data Generation & State Space ReconstructionI simulated synthetic datasets representing coupled and uncoupled pendulums. To reconstruct the phase space of these dynamical systems from 1D time-series data, I utilized **Takens’ Embedding Theorem**, which allows for the recovery of the system's underlying attractor.
2. Topological Feature ExtractionUsing giotto-tda, I applied Vietoris-Rips filtration to the embeddings.

  * **Persistence Diagrams (PD)**: I computed 1st and 2nd degree homology ($H_1, H_2$) to capture the birth and death of loops and voids within the data structure.

  * **Persistence Landscapes (PL)**: To make these topological summaries compatible with ML algorithms, I mapped the PDs into Persistence Landscapes, a stable, Hilbert space representation of the topology.
  
3. Dimensionality Reduction & Classification

* To handle the high-dimensional nature of the landscapes, I projected the features into a low-dimensional space using Principal Component Analysis (PCA).

* Support Vector Machine (SVM): I trained an SVM to identify the optimal separating hyperplane between the coupled and uncoupled states.

* Validation: I performed k-fold cross-validation, which confirmed the high classification accuracy and the robustness of topological features against variance in the oscillators' initial conditions.

# 🛠️ Technical Stack

TDA Library: giotto-tda (high-performance C++ backend)
ML Framework: scikit-learn (PCA, SVM, Cross-Validation)
Data Science: numpy, pandas, scipyVisualization: plotly (interactive 3D plots), matplotlib
Environment: conda (environment.yml included for reproducibility)

# 📈 Performance & Results

By utilizing Persistence Landscapes, the model achieved high accuracy in distinguishing coupled from uncoupled oscillators. The topological approach proved significantly more robust to noise than standard correlation-based metrics, as it captures the underlying geometry of the dynamical system.

I looked at coupled and uncoupled oscillators and use topological data analysis (TDA) and machine learning (ML) to identify the existence of coupling force. 

I generated synthetic data of coupled and uncoupled pendulums. Using Taken's embedding, I applied 1st and 2nd degree homology to the embedding of the time series data, and obtained topological summaries of the datasets, called persistence diagrams (PD). I map PDs into vector summaries persistence landscapes and project them to low dimensional vector space using principal component analysis (PCA). Then I applied support vector machine (SVM) to the vectors and find a separating hyperplane between the coupled and uncoupled oscillators. Cross validation was also performed and it confirmed the classification accuracy.

# Reproduce the experiment

To replicate the environment: conda env create -f env.yml conda activate tda-coupled-oscillators

For interactive 3D visualizations, ensure the ipympl backend is activated by including the following in the notebook: %matplotlib widget

#from persim import plot_landscape
from gtda.diagrams import PersistenceLandscape

def compute_pl(persistence_diagrams):
    '''
    Computes persistence landscapes and flattens them for ML pipelines.
    '''
    # 1. Initialize the landscape transformer
    # n_layers=15 means we track the top 15 peaks for each homology dimension
    pl_transformer = PersistenceLandscape(n_layers=15, n_bins=50, n_jobs=1)

    # 2. Transform ALL diagrams at once
    # Output shape: (n_samples, n_homology_dims * n_layers, n_bins)
    landscapes = pl_transformer.fit_transform(persistence_diagrams)

    # 3. Vectorization (The "Machine Learning" input)
    # Instead of manual concatenation, we flatten the layers and bins
    # into a single vector per sample.
    # Resulting shape: (n_samples, n_features)
    n_samples = landscapes.shape[0]
    v_coupled = landscapes.reshape(n_samples, -1)

    return v_coupled
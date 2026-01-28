#from PIL import Image
#import pandas as pd
#from pathlib import Path
#import persim
#from persim import plot_landscape
#import ipywidgets as widgets
from gtda.homology import VietorisRipsPersistence
#from gtda.diagrams import PersistenceLandscape
#from gtda.plotting import plot_point_cloud, plot_diagram
#from gtda.pipeline import Pipeline
import numpy as np
from gtda.homology import VietorisRipsPersistence

def compute_pd(data_embedded, homology_dimensions=[0, 1]):
    """
    Computes persistence diagrams for a collection of point clouds.
    data_embedded: Should be a list of 2D arrays OR a 3D array of shape (n_samples, n_points, n_dimensions)
    """
    # 1. Convert list of point clouds into a single 3D numpy array
    # Shape: (number_of_point_clouds, points_per_cloud, dimensions)
    data_stack = np.stack(data_embedded)

    # 2. Initialize the transformer
    periodic_persistence = VietorisRipsPersistence(
        homology_dimensions=homology_dimensions,
        n_jobs=-1
    )

    # 3. Process EVERYTHING at once (No for-loop needed!)
    # This is where n_jobs=6 actually gets utilized properly
    persistence_diagrams = periodic_persistence.fit_transform(data_stack)

    return persistence_diagrams
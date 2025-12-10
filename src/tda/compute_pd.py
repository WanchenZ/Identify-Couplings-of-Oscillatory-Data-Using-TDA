#from PIL import Image
#import pandas as pd
#from pathlib import Path
#import persim
from persim import plot_landscape
#import ipywidgets as widgets
from gtda.homology import VietorisRipsPersistence
#from gtda.diagrams import PersistenceLandscape
#from gtda.plotting import plot_point_cloud, plot_diagram
#from gtda.pipeline import Pipeline

def compute_pd(data_embedded, homology_dimensions = [0,1,2]):
    '''
    compute the topological summaries, persistence diagrams, for each embedded point cloud
    homology_dimensions: homology degree
    '''
    periodic_persistence_diagrams = []
    #homology_dimensions = [0,1,2]
    periodic_persistence = VietorisRipsPersistence(homology_dimensions=homology_dimensions, n_jobs=6)

    for k in range(1,50):
        data_embedded[k-1] = data_embedded[k-1][None, :, :]
        periodic_persistence_diagrams.append(periodic_persistence.fit_transform(data_embedded[k-1]))

    return periodic_persistence_diagrams
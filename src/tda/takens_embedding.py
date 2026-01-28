from gtda.time_series import SingleTakensEmbedding

def takens_embedding(data_combined, N, embedding_dimension = 6, embedding_time_delay = 5, stride = 2):
    '''
    Compute taken's embedding with window size = embedding dimension
    '''
#from PIL import Image
#import pandas as pd
#from pathlib import Path
#import persim
#from persim import plot_landscape
#import ipywidgets as widgets
#from gtda.homology import VietorisRipsPersistence
#from gtda.diagrams import PersistenceLandscape
#from gtda.plotting import plot_point_cloud, plot_diagram
#from gtda.pipeline import Pipeline
    data_embedded = []
    #embedding_dimension_periodic = 6
    #embedding_time_delay_periodic = 5
    #stride = 2
    STE=SingleTakensEmbedding(parameters_type="fixed", time_delay=embedding_time_delay,
        dimension=embedding_dimension,stride=stride)

    for k in range(N):
        data_embedded.append(STE.fit_transform(data_combined[k]))
        
    return data_embedded
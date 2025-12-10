from persim import plot_landscape
from gtda.diagrams import PersistenceLandscape

def compute_pl(periodic_persistence_diagrams):
    '''
    compute the vector summaries for plugging into machine learning pipelines and doing statistical analysis
    '''


    pl = PersistenceLandscape(n_layers=15, n_bins=50, n_jobs=None)

    PersL = []
    v_coupled = []

    for k in range(1,50):
        PersL.append(pl.fit_transform_plot(periodic_persistence_diagrams[k-1]))# form vector input for SVM using H1 and H2
    # use first 15 layers in H1 and H2
        v_coupled.append(np.concatenate((PersL[k-1][0,15],PersL[k-1][0,16], PersL[k-1][0,17] , PersL[k-1][0,18] , PersL[k-1][0,19],
                                    PersL[k-1][0,20] , PersL[k-1][0,21] ,PersL[k-1][0,22] ,PersL[k-1][0,23] ,PersL[k-1][0,24] ,
                                   PersL[k-1][0,25] ,PersL[k-1][0,26] ,PersL[k-1][0,27] ,PersL[k-1][0,28] ,PersL[k-1][0,29] ,
                                   PersL[k-1][0,30] ,PersL[k-1][0,31] ,PersL[k-1][0,32] ,PersL[k-1][0,33] ,PersL[k-1][0,34] ,
                                   PersL[k-1][0,35] ,PersL[k-1][0,36] ,PersL[k-1][0,37] ,PersL[k-1][0,38] ,PersL[k-1][0,39] ,
                                   PersL[k-1][0,40] ,PersL[k-1][0,41] ,PersL[k-1][0,42] , PersL[k-1][0,43] ,PersL[k-1][0,44]), axis = 0))

    return v_coupled
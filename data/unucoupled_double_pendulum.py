import math
import numpy as np
import random

def generate_coupled_pendulum(N = 50, T = 30, spacing = 20):
  '''
  Generate data of double pendulum with different coupling constants
  N: number of datasets
  T: period
  spacing: resolution
  '''
  L = 1
  I = L * L /9.81
  w_0 = math.sqrt(L/I)
#T = 30
#spacing = 20
  t = np.linspace(0, T, T*spacing)
  w = []
  theta_uncoupled = []
  phi_uncoupled = []
  data_combined_uncoupled = []
#data_embedded_uncoupled = []
#periodic_persistence_diagrams_uncoupled = []
#PersL_uncoupled = []
  #v_uncoupled = []
#pl = PersistenceLandscape(n_layers=15, n_bins=50, n_jobs=None)

  for k in range(1,N):
    w_1 = random.uniform(2.5, 8.5)
    w_2 = random.uniform(2.5, 8.5)
    phase = random.uniform(0, 6.28)
  #w.append(math.sqrt((L + 1/2 * k_0 * L * L) / I ))
    theta_uncoupled.append(np.cos( w_1 * t ))
    phi_uncoupled.append(np.cos( w_2 * t + phase))
  #w[k-1] = math.sqrt((L + 1/2 * k * L * L) / I )
  #theta[k-1] = np.cos( (w_0- w[k])/2 * t ) * np.cos( (w_0+ w[k])/2 * t ) #+ np.random.normal(0, 0.1, len(t)) # standard deviation is a tenth of the amplitude
  #phi[k-1] = np.sin( (w_0-w[k])/2 * t ) * np.sin( (w_0+w[k])/2 * t ) # + np.random.normal(0, 0.1, len(t))
    theta_uncoupled[k-1] = np.array(theta_uncoupled[k-1])
    phi_uncoupled[k-1] = np.array(phi_uncoupled[k-1])
    data_combined_uncoupled.append(np.empty(T*spacing*2))

    for i in range(T*spacing):
      data_combined_uncoupled[k-1][2*i] = theta_uncoupled[k-1][i]
      data_combined_uncoupled[k-1][2*i+1] = phi_uncoupled[k-1][i]

  return data_combined_uncoupled, N
#  data_embedded_uncoupled.append(STE.fit_transform(data_combined_uncoupled[k-1]))
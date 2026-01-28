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
  w_0 = math.sqrt(L/I) # L/I = 1
  #T = 30
  #spacing = 20
  t = np.linspace(0, T, T*spacing)
  w = []
  theta = []
  phi = []

# concatenate into 1d array
  data_combined = []

# taken's embedding
#data_embedded = []

  for k in range(1,N+1):
    w.append(math.sqrt((L + 1/2 * k * L * L) / I ))
    theta.append(np.cos( (w_0- w[k-1])/2 * t ) * np.cos( (w_0+ w[k-1])/2 * t ))
    phi.append(np.sin( (w_0-w[k-1])/2 * t ) * np.sin( (w_0+w[k-1])/2 * t ))
    #w[k-1] = math.sqrt((L + 1/2 * k * L * L) / I )
    #theta[k-1] = np.cos( (w_0- w[k])/2 * t ) * np.cos( (w_0+ w[k])/2 * t ) #+ np.random.normal(0, 0.1, len(t)) # standard deviation is a tenth of the amplitude
    #phi[k-1] = np.sin( (w_0-w[k])/2 * t ) * np.sin( (w_0+w[k])/2 * t ) # + np.random.normal(0, 0.1, len(t))
    theta[k-1] = np.array(theta[k-1])
    phi[k-1] = np.array(phi[k-1])
    data_combined.append(np.empty(T*spacing*2))

    for i in range(T*spacing):
      data_combined[k-1][2*i] = theta[k-1][i]
      data_combined[k-1][2*i+1] = phi[k-1][i]

  #data_embedded.append(STE.fit_transform(data_combined[k-1]))
  return data_combined, theta, phi
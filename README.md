# Identify-Couplings-of-Oscillatory-Data-Using-TDA

I looked at coupled and uncoupled oscillators and use topological data analysis (TDA) and machine learning (ML) to identify the existence of coupling force. 

I generated synthetic data of coupled and uncoupled pendulums. Using Taken's embedding, I applied 1st and 2nd degree homology to the embedding of the time series data, and obtained topological summaries of the datasets, called persistence diagrams (PD). I map PDs into vector summaries persistence landscapes and project them to low dimensional vector space using principal component analysis (PCA). Then I applied support vector machine (SVM) to the vectors and find a separating hyperplane between the coupled and uncoupled oscillators. Cross validation was also performed and it confirmed the classification accuracy.

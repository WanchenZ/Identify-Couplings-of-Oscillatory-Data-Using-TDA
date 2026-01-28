

'''
Reduce the dimension using Principal Component Analysis
Then apply Support Vector Machine with 20% training 80% testing
Cross validation
'''# more functions

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def test_on_svm(v_coupled, v_uncoupled):
    # 1. Dynamic label creation (Prevents the ValueError)
    X = np.concatenate((np.array(v_coupled), np.array(v_uncoupled)))
    
    y = np.concatenate((
        np.ones(len(v_coupled)), 
        np.zeros(len(v_uncoupled))
    ))

    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

    # 2. PCA to 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    # 4. Train and Cross-Validate
    clf = SVC(kernel='linear')
    cv_scores = cross_val_score(clf, X_pca, y, cv=5)
    print(f"Average CV Accuracy: {np.mean(cv_scores):.2f}")

    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"Test Accuracy: {accuracy:.2f}")

    # 5. Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Coupled (1) vs Uncoupled (0)
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], X_pca[y==1, 2], 
               color='blue', label='Coupled', alpha=0.6)
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], X_pca[y==0, 2], 
               color='red', label='Uncoupled', alpha=0.6)

    # 6. Plot the 3D Decision Hyperplane (Linear)
    # The equation is: w0*x + w1*y + w2*z + b = 0
    # So: z = -(w0*x + w1*y + b) / w2
    tmp_x = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 10)
    tmp_y = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 10)
    xx, yy = np.meshgrid(tmp_x, tmp_y)
    
    w = clf.coef_[0]
    b = clf.intercept_[0]
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]

    ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')

    ax.set_title('TDA Features: SVM Linear Separation (PCA Space)')
    ax.legend()
    plt.show()

    return accuracy, cv_scores
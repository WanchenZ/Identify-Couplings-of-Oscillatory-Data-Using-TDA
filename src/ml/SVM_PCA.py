

'''
Reduce the dimension using Principal Component Analysis
Then apply Support Vector Machine with 20% training 80% testing
Cross validation
'''
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score

def test_on_svm(v_coupled, v_uncoupled):

# Concatenate the datasets
    X = np.concatenate((np.array(v_coupled), np.array(v_uncoupled)))
    y = np.concatenate((np.ones(49), np.zeros(49)))

# Apply PCA to reduce dimensionality to 3
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train an SVM classifier
    clf = SVC(kernel='linear')

# Perform cross-validation
    cv_scores = cross_val_score(clf, X_pca, y, cv=5)

# Output the cross-validation scores
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Average Accuracy: {np.mean(cv_scores):.2f} (± {np.std(cv_scores):.2f})")

# Fit the classifier on the training set
    clf.fit(X_train, y_train)

# Make predictions on the test set
    y_pred = clf.predict(X_test)

# Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

# Plot the 3D decision boundary
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

# Plot the points in 3D
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap=plt.cm.Paired, s=50)

# Plot the training points in 3D
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap=plt.cm.Paired, marker='x', s=100, label='Training Points')

# Plot the decision surface
    xx, yy = np.meshgrid(np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100),
                     np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Set labels
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('SVM Decision Boundary in 3D')

    plt.show()

    return accuracy, cv_scores
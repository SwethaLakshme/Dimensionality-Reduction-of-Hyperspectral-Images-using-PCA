from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io as sio
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # load the dataset
    data = sio.loadmat('PaviaU.mat')['paviaU']

    # extract the spectral data
    spectral_data = np.reshape(data, (-1, data.shape[-1]))

    # calculate the covariance matrix
    covariance_matrix = np.cov(spectral_data.T)

    # calculate the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # sort the eigenvectors in descending order based on their corresponding eigenvalues
    eigenvectors_sorted = eigenvectors[:, eigenvalues.argsort()[::-1]]

    # choose the top k eigenvectors as the principal components
    k = 30
    principal_components = eigenvectors_sorted[:, :k]

    # project the original data onto the new feature space
    reduced_data = np.dot(spectral_data, principal_components)

    # load the ground truth data
    gt_data = sio.loadmat('PaviaU_gt.mat')['paviaU_gt']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(reduced_data, gt_data.ravel(), test_size=0.3, random_state=42)

    # perform PCA on the testing data using the same eigenvectors as the training data
    data_test = sio.loadmat('PaviaU.mat')['paviaU']
    spectral_data_test = np.reshape(data_test, (-1, data_test.shape[-1]))
    mean_data_test = np.mean(spectral_data_test, axis=0)
    centered_data_test = spectral_data_test - mean_data_test
    reduced_data_test = np.dot(centered_data_test, principal_components)

    # train the k-nearest neighbors classifier on the original data
    # knn_original = KNeighborsClassifier(n_neighbors=5)
    # knn_original.fit(spectral_data, gt_data.ravel())

    # train the k-nearest neighbors classifier on the reduced data
    knn_reduced = KNeighborsClassifier(n_neighbors=5)
    knn_reduced.fit(X_train, y_train)

    # evaluate the accuracy of the classifiers on the testing data
    # y_pred_original = knn_original.predict(spectral_data_test)
    y_pred_reduced = knn_reduced.predict(reduced_data_test)

    # accuracy_original = accuracy_score(gt_data.ravel(), y_pred_original)
    accuracy_reduced = accuracy_score(gt_data.ravel(), y_pred_reduced)

    return render_template('result.html',reduced=accuracy_reduced)
if __name__ == "__main__":
    app.run("localhost", "9999", debug=True)
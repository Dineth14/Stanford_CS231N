# K-Nearest Neighbors (KNN) Classifier for CIFAR-10 Dataset

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tarfile
import urllib.request
from sklearn.metrics import accuracy_score


class KNNClassifier(object):
    def __init__(self, k=5) -> None:
        self.k = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            if (i + 1) % 50 == 0 or i == 0:
                print(f'  Predicting {i + 1}/{num_test}...')
            distances = np.linalg.norm(self.X_train - X[i], axis=1)
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            y_pred[i] = np.bincount(nearest_labels).argmax()

        return y_pred

def load_cifar10():
    """Download CIFAR-10 from the official source and return train/test splits."""
    data_dir = os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py')
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    archive = os.path.join(os.path.dirname(__file__), 'cifar-10-python.tar.gz')

    if not os.path.isdir(data_dir):
        print('Downloading CIFAR-10 dataset...')
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(__file__))
        os.remove(archive)
        print('Download complete.')

    def _unpickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    # Load training batches 1-5
    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch = _unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        X_train_list.append(batch[b'data'])
        y_train_list.append(batch[b'labels'])
    X_train = np.concatenate(X_train_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array([l for sublist in y_train_list for l in sublist], dtype=np.int64)

    # Load test batch
    test_batch = _unpickle(os.path.join(data_dir, 'test_batch'))
    X_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_batch[b'labels'], dtype=np.int64)

    return (X_train, y_train), (X_test, y_test)

def plot_sample_images(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(X))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
        plt.xlabel(y[i])
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(y_true, y_pred, show_plots=False):
    from sklearn.metrics import classification_report

    print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(classification_report(y_true, y_pred))
    if show_plots:
        plot_confusion_matrix(y_true, y_pred)



def visualize_predictions(X, y_true, y_pred):
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(X))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        plt.xlabel(f'True: {y_true[i]}, Pred: {y_pred[i]}', color=color)
    plt.show()

def main():
    show_plots = False

    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = load_cifar10()

    # Use a subset for reasonable runtime (KNN is O(n) per query)
    X_train, y_train = X_train[:3000], y_train[:3000]
    X_test, y_test = X_test[:300], y_test[:300]

    X_test_images = X_test.copy()

    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Create and train KNN classifier
    knn = KNNClassifier()
    knn.train(X_train, y_train)

    # Make predictions on test set
    y_pred = knn.predict(X_test)

    # Evaluate accuracy
    evaluate_model(y_test, y_pred, show_plots=show_plots)

    if show_plots:
        # Visualize predictions with original image tensors.
        visualize_predictions(X_test_images, y_test, y_pred)


if __name__ == '__main__':
    main()


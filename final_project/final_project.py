import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#load data
def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y
#normalization
def normalize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

#split the training set
def _train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets with shuffled data.

    Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Labels of shape (n_samples,).
        test_size (float): Proportion of the dataset to include in the test split (default=0.2).
        random_state (int): Seed for reproducibility (default=None).

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Shuffle data
    X = X[indices]
    y = y[indices]
    
    # Split data
    n_samples = len(X)
    test_size = int(n_samples * test_size)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    return X_train, X_test, y_train, y_test

def _leaky_relu(z):
    return np.where(z > 0, z, 0.01 * z)  # Leaky ReLU

def _leaky_relu_derivative(z):
    return np.where(z > 0, 1, 0.01)
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent spillage
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


1.#K++
class Kpp:
    def initialize_centroids_kmeans_pp(self,X, k):
        """
        Initialize centroids using K-Means++ algorithm.
        The first centroid is randomly selected, and then the subsequent 
        centroids are chosen based on the distance between the existing centroids and the data points,
        according to a probability distribution based on the distances.
        Parameters:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            k (int): Number of clusters.
        Returns:
            centroids (numpy.ndarray): Initialized centroids of shape (k, n_features).
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((k, n_features))
        centroids[0] = X[np.random.randint(n_samples)]  # First centroid randomly chosen

        for i in range(1, k):
            distances = np.min([np.linalg.norm(X - centroid, axis=1)**2 for centroid in centroids[:i]], axis=0)
            probabilities = distances / distances.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[i] = X[j]
                    break
        return centroids

    def kmeans(self,X, k, max_iters=300, tol=1e-4):
        """
        Perform K-Means clustering.
        Parameters:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            k (int): Number of clusters.
            max_iters (int): Maximum number of iterations.
            tol (float): Tolerance for centroid change.
        Returns:
            centroids (numpy.ndarray): Final centroids of shape (k, n_features).
            labels (numpy.ndarray): Cluster assignments of shape (n_samples,).
        """
        centroids = self.initialize_centroids_kmeans_pp(X, k)
        prev_centroids = np.zeros_like(centroids)
        labels = np.zeros(X.shape[0])

        for iteration in range(max_iters):
            # Assign clusters
            for i, sample in enumerate(X):
                distances = np.linalg.norm(sample - centroids, axis=1)
                labels[i] = np.argmin(distances)

            # Update centroids
            for cluster_idx in range(k):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) > 0:
                    centroids[cluster_idx] = cluster_points.mean(axis=0)

            # Check for convergence
            if np.linalg.norm(centroids - prev_centroids) < tol:
                break
            prev_centroids = centroids.copy()

        return centroids, labels
    # Visualization Module
    def visualize_kmeans(self,X, centroids, labels, title="K-Means++ Clustering"):
        """
        Visualize the results of K-Means clustering.

        Parameters:
            X (numpy.ndarray): Data points of shape (n_samples, n_features).
            centroids (numpy.ndarray): Cluster centroids of shape (k, n_features).
            labels (numpy.ndarray): Cluster assignments of shape (n_samples,).
            title (str): Title of the plot.
        """
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
        plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x')
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

2.#soft K++
class soft_Kpp:
    def soft_kmeans(self,X, k, beta=1.0, max_iters=300, tol=1e-4):
        """
        Perform Soft K-Means clustering.
        Parameters:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            k (int): Number of clusters.
            beta (float): Softness parameter (inverse temperature).
            max_iters (int): Maximum number of iterations.
            tol (float): Tolerance for centroid change.
        Returns:
            centroids (numpy.ndarray): Final centroids of shape (k, n_features).
            probabilities (numpy.ndarray): Soft assignments of shape (n_samples, k).
        """
        n_samples, n_features = X.shape
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        prev_centroids = np.zeros_like(centroids)
        probabilities = np.zeros((n_samples, k))

        for iteration in range(max_iters):
            # Compute probabilities using softmax
            distances = np.array([[np.linalg.norm(sample - centroid) for centroid in centroids] for sample in X])
            probabilities = np.exp(-beta * distances)
            probabilities /= probabilities.sum(axis=1, keepdims=True)

            # Update centroids using weighted averages
            for j in range(k):
                weights = probabilities[:, j].reshape(-1, 1)
                centroids[j] = np.sum(weights * X, axis=0) / weights.sum()

            # Check for convergence
            if np.linalg.norm(centroids - prev_centroids) < tol:
                break
            prev_centroids = centroids.copy()

        return centroids, probabilities
    # Visualization Module
    def visualize_soft_kmeans(self,X, centroids, probabilities, title="Soft K-Means Clustering"):
        """
        Visualize the results of Soft K-Means clustering.

        Parameters:
            X (numpy.ndarray): Data points of shape (n_samples, n_features).
            centroids (numpy.ndarray): Cluster centroids of shape (k, n_features).
            probabilities (numpy.ndarray): Soft assignments of shape (n_samples, k).
            title (str): Title of the plot.
        """
        soft_labels = np.argmax(probabilities, axis=1)
        plt.scatter(X[:, 0], X[:, 1], c=soft_labels, cmap='viridis', s=50)
        plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x')
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()


#3.PCA
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.components_ = eigenvectors[:, :self.n_components]
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def plot_principal_components(self,X_transformed, labels, n_components):
        if n_components == 2:
                plt.figure(figsize=(8, 6))
                for label in np.unique(labels):
                    plt.scatter(X_transformed[labels == label, 0], X_transformed[labels == label, 1], label=f"Class {label}")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.title("2D PCA Visualization")
                plt.legend()
                plt.show()

        elif n_components == 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                for label in np.unique(labels):
                    ax.scatter(
                        X_transformed[labels == label, 0],
                        X_transformed[labels == label, 1],
                        X_transformed[labels == label, 2],
                        label=f"Class {label}"
                    )
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.set_zlabel("Principal Component 3")
                ax.set_title("3D PCA Visualization")
                plt.legend()
                plt.show()

#4.非线性编码器
class NonlinearAutoEncoder:
    def __init__(self, input_dim, latent_dim, hidden_dims=[128, 64], learning_rate=0.01):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.weights, self.biases = self._initialize_weights()

    def _initialize_weights(self):
        layers = [self.input_dim] + self.hidden_dims + [self.latent_dim] + self.hidden_dims[::-1] + [self.input_dim]
        weights = {}
        biases = {}
        for i in range(len(layers) - 1):
            limit = np.sqrt(2 / layers[i])  # He initialization
            weights[f"W{i+1}"] = np.random.uniform(-limit, limit, (layers[i], layers[i+1]))
            biases[f"b{i+1}"] = np.zeros((1, layers[i+1]))
        return weights, biases

    def _forward_pass(self, X):
        activations = {"A0": X}
        linear_combinations = {}
        for i in range(1, len(self.weights) + 1):
            z = np.dot(activations[f"A{i-1}"], self.weights[f"W{i}"]) + self.biases[f"b{i}"]
            linear_combinations[f"Z{i}"] = z
            activations[f"A{i}"] = _leaky_relu(z) if i < len(self.weights) else z
        return activations, linear_combinations

    def _backward_pass(self, X, activations, linear_combinations):
        gradients = {}
        m = X.shape[0]
        A_final = activations[f"A{len(self.weights)}"]
        dA = A_final - X
        for i in reversed(range(1, len(self.weights) + 1)):
            dZ = dA if i == len(self.weights) else dA * _leaky_relu_derivative(linear_combinations[f"Z{i}"])
            gradients[f"dW{i}"] = np.dot(activations[f"A{i-1}"].T, dZ) / m
            gradients[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
            if i > 1:
                dA = np.dot(dZ, self.weights[f"W{i}"].T)
        return gradients

    def _update_parameters(self, gradients):
        for i in range(1, len(self.weights) + 1):
            self.weights[f"W{i}"] -= self.learning_rate * gradients[f"dW{i}"]
            self.biases[f"b{i}"] -= self.learning_rate * gradients[f"db{i}"]

    def train(self, X, epochs=50, batch_size=32):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                activations, linear_combinations = self._forward_pass(X_batch)
                gradients = self._backward_pass(X_batch, activations, linear_combinations)
                self._update_parameters(gradients)
            activations, _ = self._forward_pass(X)
            loss = np.mean((activations[f"A{len(self.weights)}"] - X) ** 2)
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    def encode(self, X):
        activations, _ = self._forward_pass(X)
        return activations[f"A{len(self.hidden_dims) + 1}"]

    def decode(self, X_latent):
        activations = {"A0": X_latent}
        for i in range(len(self.hidden_dims) + 2, len(self.weights) + 1):
            z = np.dot(activations[f"A{i-1}"], self.weights[f"W{i}"]) + self.biases[f"b{i}"]
            activations[f"A{i}"] = _leaky_relu(z) if i < len(self.weights) else z
        return activations[f"A{len(self.weights)}"]
    def plot_encoded_data(self,X_encoded, labels, n_components):
            if n_components == 2:
                plt.figure(figsize=(8, 6))
                for label in np.unique(labels):
                    plt.scatter(X_encoded[labels == label, 0], X_encoded[labels == label, 1], label=f"Class {label}")
                plt.xlabel("Latent Dimension 1")
                plt.ylabel("Latent Dimension 2")
                plt.title("2D Latent Space Visualization")
                plt.legend()
                plt.show()

            elif n_components == 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                for label in np.unique(labels):
                    ax.scatter(
                        X_encoded[labels == label, 0],
                        X_encoded[labels == label, 1],
                        X_encoded[labels == label, 2],
                        label=f"Class {label}"
                    )
                ax.set_xlabel("Latent Dimension 1")
                ax.set_ylabel("Latent Dimension 2")
                ax.set_zlabel("Latent Dimension 3")
                ax.set_title("3D Latent Space Visualization")
                plt.legend()
                plt.show()

#5.Clustering with Reduced Dimensions
class Clustering:
    def plot_clusters(self,X_reduced, labels, n_components, title):
        if n_components == 2:
            plt.figure(figsize=(8, 6))
            for cluster in np.unique(labels):
                plt.scatter(
                    X_reduced[labels == cluster, 0],
                    X_reduced[labels == cluster, 1],
                    label=f"Cluster {int(cluster)}"
                )
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.title(title)
            plt.legend()
            plt.show()
        elif n_components == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for cluster in np.unique(labels):
                ax.scatter(
                    X_reduced[labels == cluster, 0],
                    X_reduced[labels == cluster, 1],
                    X_reduced[labels == cluster, 2],
                    label=f"Cluster {int(cluster)}"
                )
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.set_zlabel("Dimension 3")
            ax.set_title(title)
            plt.legend()
            plt.show()
    #Cluster alignment
    def map_cluster_to_labels(self,labels, y_true):
        """
        Map clustering labels to ground truth labels using maximum assignment.

        Parameters:
            labels (numpy.ndarray): Clustering labels of shape (n_samples,).
            y_true (numpy.ndarray): True labels of shape (n_samples,).

        Returns:
            mapped_labels (numpy.ndarray): Remapped clustering labels aligned with true labels.
        """
        unique_clusters = np.unique(labels)
        unique_classes = np.unique(y_true)

        # Create a cost matrix where the value is the count of matching pairs
        cost_matrix = np.zeros((len(unique_clusters), len(unique_classes)))

        for i, cluster in enumerate(unique_clusters):
            for j, true_class in enumerate(unique_classes):
                cost_matrix[i, j] = np.sum((labels == cluster) & (y_true == true_class))

        # Compute mapping by maximizing assignment
        mapping = {}
        for cluster_idx in range(len(unique_clusters)):
            max_class_idx = np.argmax(cost_matrix[cluster_idx])
            mapping[unique_clusters[cluster_idx]] = unique_classes[max_class_idx]

        # Map labels using the computed mapping
        mapped_labels = np.array([mapping[label] for label in labels])
        return mapped_labels
    #Categorical metric calculations
    def calculate_metrics(self,y_true, y_pred):
        """
        Calculate Accuracy, Precision, Recall, and F1-score.

        Parameters:
            y_true (numpy.ndarray): True labels of shape (n_samples,).
            y_pred (numpy.ndarray): Predicted labels of shape (n_samples,).

        Returns:
            metrics (dict): Dictionary containing Accuracy, Precision, Recall, and F1-score.
        """
        unique_classes = np.unique(y_true)
        true_positives = np.zeros(len(unique_classes))
        false_positives = np.zeros(len(unique_classes))
        false_negatives = np.zeros(len(unique_classes))

        for i, cls in enumerate(unique_classes):
            true_positives[i] = np.sum((y_pred == cls) & (y_true == cls))
            false_positives[i] = np.sum((y_pred == cls) & (y_true != cls))
            false_negatives[i] = np.sum((y_pred != cls) & (y_true == cls))

        accuracy = np.sum(true_positives) / len(y_true)
        precision = np.sum(true_positives / (true_positives + false_positives + 1e-10)) / len(unique_classes)
        recall = np.sum(true_positives / (true_positives + false_negatives + 1e-10)) / len(unique_classes)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        }
    
#6.mlp
class MLP:
    def preprocess_labels(self,labels, num_classes):
        one_hot_labels = np.zeros((labels.size, num_classes))
        one_hot_labels[np.arange(labels.size), labels - 1] = 1  # Assuming labels are 1, 2, 3
        return one_hot_labels
    def softmax(self,x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self,y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def initialize_weights(self,input_dim, hidden_dim, output_dim):
        weights = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.01,
            'b1': np.zeros((1, hidden_dim)),
            'W2': np.random.randn(hidden_dim, output_dim) * 0.01,
            'b2': np.zeros((1, output_dim))
        }
        return weights
    def forward_propagation(self,X, weights):
        Z1 = np.dot(X, weights['W1']) + weights['b1']
        A1 = np.tanh(Z1)
        Z2 = np.dot(A1, weights['W2']) + weights['b2']
        A2 = self.softmax(Z2)
        activations = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return activations
    def backward_propagation(self,X, y_true, activations, weights):
        m = X.shape[0]
        dZ2 = activations['A2'] - y_true
        dW2 = np.dot(activations['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, weights['W2'].T)
        dZ1 = dA1 * (1 - np.power(activations['A1'], 2))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return gradients
    def update_weights(self,weights, gradients, learning_rate):
        weights['W1'] -= learning_rate * gradients['dW1']
        weights['b1'] -= learning_rate * gradients['db1']
        weights['W2'] -= learning_rate * gradients['dW2']
        weights['b2'] -= learning_rate * gradients['db2']
    def train_mlp(self,X_train, y_train, input_dim, hidden_dim, output_dim, epochs, learning_rate):
        weights = self.initialize_weights(input_dim, hidden_dim, output_dim)
        for epoch in range(epochs):
            activations = self.forward_propagation(X_train, weights)
            loss = self.cross_entropy_loss(y_train, activations['A2'])
            gradients = self.backward_propagation(X_train, y_train, activations, weights)
            self.update_weights(weights, gradients, learning_rate)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        return weights
    def predict(self,X, weights):
        activations = self.forward_propagation(X, weights)
        return np.argmax(activations['A2'], axis=1) + 1  # Convert to label range (1, 2, 3)

#7.svm
# Linear SVM for Multi-class Classification using One-vs-Rest
class SVM:
    def train_linear_svm_multi(self,X_train, y_train, learning_rate=0.001, epochs=1000, C=0.01):
        """
        Train multiple binary SVM classifiers (One-vs-Rest) for multi-class classification.
        Parameters:
            X_train (numpy.ndarray): Training feature matrix of shape (n_samples, n_features).
            y_train (numpy.ndarray): Training labels of shape (n_samples,).
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of iterations for gradient descent.
            C (float): Regularization parameter.
        Returns:
            classifiers (list): List of (w, b) for each binary classifier.
        """
        n_samples, n_features = X_train.shape
        classes = np.unique(y_train)
        classifiers = []

        for c in classes:
            # Convert y_train to binary for current class
            y_binary = np.where(y_train == c, 1, -1)
            w = np.zeros(n_features)
            b = 0

            for epoch in range(epochs):
                for i in range(n_samples):
                    condition = (y_binary[i] * (np.dot(X_train[i], w) + b)) >= 1
                    if condition:
                        w -= learning_rate * (2 * C * w)
                    else:
                        w -= learning_rate * (2 * C * w - X_train[i] * y_binary[i])
                        b -= learning_rate * y_binary[i]

            classifiers.append((w, b))

        return classifiers

    def predict_linear_svm_multi(self,X_test, classifiers):
        """
        Predict multi-class labels using multiple binary SVM classifiers.
        Parameters:
            X_test (numpy.ndarray): Testing feature matrix of shape (n_samples, n_features).
            classifiers (list): List of (w, b) for each binary classifier.
        Returns:
            y_pred (numpy.ndarray): Predicted labels of shape (n_samples,).
        """
        n_samples = X_test.shape[0]
        scores = np.zeros((n_samples, len(classifiers)))

        for idx, (w, b) in enumerate(classifiers):
            scores[:, idx] = np.dot(X_test, w) + b

        return np.argmax(scores, axis=1) + 1  # Classes are 1-indexed

    # Gaussian Kernel SVM for Multi-class Classification using One-vs-Rest
    def rbf_kernel(self,x1, x2, gamma):
        return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

    def train_gaussian_svm_multi(self,X_train, y_train, gamma=0.5, C=1.0, epochs=1000):
        """
        Train multiple Gaussian Kernel SVM classifiers (One-vs-Rest) for multi-class classification.
        """
        n_samples = X_train.shape[0]
        classes = np.unique(y_train)
        classifiers = []

        for c in classes:
            y_binary = np.where(y_train == c, 1, -1)
            alpha = np.zeros(n_samples)
            K = np.array([[self.rbf_kernel(x1, x2, gamma) for x2 in X_train] for x1 in X_train])

            for _ in range(epochs):
                for i in range(n_samples):
                    gradient = 1 - np.sum(alpha * y_binary * K[:, i])
                    alpha[i] += 0.001 * gradient
                    alpha[i] = np.clip(alpha[i], 0, C)

            classifiers.append((alpha, y_binary, K, gamma))

        return classifiers

    def predict_gaussian_svm_multi(self,X_train, X_test, classifiers):
        """
        Predict multi-class labels using multiple Gaussian Kernel SVM classifiers.
        """
        n_samples = X_test.shape[0]
        scores = np.zeros((n_samples, len(classifiers)))

        for idx, (alpha, y_binary, K_train, gamma) in enumerate(classifiers):
            for i, x_test in enumerate(X_test):
                scores[i, idx] = np.sum(alpha * y_binary * np.array([self.rbf_kernel(x_train, x_test, gamma) for x_train in X_train]))

        return np.argmax(scores, axis=1) + 1  # Classes are 1-indexed

def evaluate_multiclass_classification(y_true, y_pred):
    """
    Evaluate multi-class classification results.

    Parameters:
        y_true (numpy.ndarray): True labels of shape (n_samples,).
        y_pred (numpy.ndarray): Predicted labels of shape (n_samples,).

    Returns:
        None. Prints evaluation metrics.
    """
    accuracy = np.mean(y_true == y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    unique_classes = np.unique(y_true)
    confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t - 1, p - 1] += 1
    print(confusion_matrix)


#8.adaboost
class Adaboost:
    def train_adaboost_multi(self,X, y, n_estimators=50):
        """
        Train AdaBoost for multi-class classification using One-vs-Rest strategy.

        Parameters:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            y (numpy.ndarray): Labels of shape (n_samples,).
            n_estimators (int): Number of weak classifiers per class.

        Returns:
            models (list): List of (classifiers, alphas) for each class.
        """
        classes = np.unique(y)
        models = []

        for c in classes:
            # Convert labels to binary for current class
            y_binary = np.where(y == c, 1, -1)

            # Train binary AdaBoost for current class
            n_samples = len(X)
            weights = np.ones(n_samples) / n_samples
            classifiers = []
            alphas = []

            for _ in range(n_estimators):
                best_feature, best_threshold, best_polarity, best_error = 0, 0, 1, float('inf')

                # Train weak classifier
                for feature in range(X.shape[1]):
                    thresholds = np.unique(X[:, feature])
                    for threshold in thresholds:
                        for polarity in [1, -1]:
                            predictions = polarity * np.sign(X[:, feature] - threshold)
                            error = np.sum(weights[predictions != y_binary])
                            if error < best_error:
                                best_feature = feature
                                best_threshold = threshold
                                best_polarity = polarity
                                best_error = error

                # Compute alpha
                alpha = 0.5 * np.log((1 - best_error) / max(best_error, 1e-10))
                predictions = best_polarity * np.sign(X[:, best_feature] - best_threshold)

                # Update weights
                weights *= np.exp(-alpha * y_binary * predictions)
                weights /= np.sum(weights)

                # Store weak classifier and alpha
                classifiers.append((best_feature, best_threshold, best_polarity))
                alphas.append(alpha)

            models.append((classifiers, alphas))

        return models


    def predict_adaboost_multi(self,X, models):
        """
        Predict labels using trained AdaBoost multi-class model.

        Parameters:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            models (list): List of (classifiers, alphas) for each class.

        Returns:
            y_pred (numpy.ndarray): Predicted labels of shape (n_samples,).
        """
        n_samples = X.shape[0]
        n_classes = len(models)
        scores = np.zeros((n_samples, n_classes))

        for class_idx, (classifiers, alphas) in enumerate(models):
            for alpha, (feature, threshold, polarity) in zip(alphas, classifiers):
                predictions = polarity * np.sign(X[:, feature] - threshold)
                scores[:, class_idx] += alpha * predictions

        return np.argmax(scores, axis=1) + 1  # Classes are 1-indexed


#9.binary classifications
def prepare_binary_data(X, y):
    mask = y != 2  # Remove class 2
    X_binary = X[mask]
    y_binary = y[mask]
    y_binary = np.where(y_binary == 1, 1, -1)  # Convert to binary labels (-1, 1)
    return X_binary, y_binary

# Step 9.1: MLP Implementation

def train_mlp_bi(X, y, input_dim, hidden_dim, learning_rate, epochs):
    """
    Simplified MLP for binary classification with fixed learning rate and ReLU activation.
    """
    weights = {
        'w1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim),
        'b1': np.zeros((1, hidden_dim)),
        'w2': np.random.randn(hidden_dim, 1) * np.sqrt(2 / hidden_dim),
        'b2': np.zeros((1, 1)),
    }

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, weights['w1']) + weights['b1']
        a1 = relu(z1)
        z2 = np.dot(a1, weights['w2']) + weights['b2']
        a2 = sigmoid(z2)

        # Compute binary cross-entropy loss
        loss = -np.mean(y * np.log(a2 + 1e-10) + (1 - y) * np.log(1 - a2 + 1e-10))

        # Backward pass
        dz2 = a2 - y.reshape(-1, 1)
        dw2 = np.dot(a1.T, dz2) / len(X)
        db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)
        dz1 = np.dot(dz2, weights['w2'].T) * relu_derivative(z1)
        dw1 = np.dot(X.T, dz1) / len(X)
        db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)

        # Update weights
        weights['w1'] -= learning_rate * dw1
        weights['b1'] -= learning_rate * db1
        weights['w2'] -= learning_rate * dw2
        weights['b2'] -= learning_rate * db2

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    return weights

def predict_mlp_bi(X, weights):
    z1 = np.dot(X, weights['w1']) + weights['b1']
    a1 = relu(z1)
    z2 = np.dot(a1, weights['w2']) + weights['b2']
    a2 = sigmoid(z2)
    return np.where(a2 > 0.5, 1, 0)  # 返回二分类标签 (0, 1)



# Step 9.2: Linear SVM
def train_linear_svm_bi(X_train, y_train, learning_rate=0.001, epochs=2000, C=1.0):
    """
    Train a linear SVM for binary classification.

    Parameters:
        X_train (numpy.ndarray): Training feature matrix of shape (n_samples, n_features).
        y_train (numpy.ndarray): Binary labels of shape (n_samples,) with values -1 and 1.
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of iterations for gradient descent.
        C (float): Regularization parameter.

    Returns:
        w (numpy.ndarray): Weight vector of shape (n_features,).
        b (float): Bias term.
    """
    n_samples, n_features = X_train.shape
    w = np.random.randn(n_features) * 0.01  # Initialize weights with small random values
    b = np.random.randn() * 0.01  # Initialize bias with a small random value

    for epoch in range(epochs):
        for i in range(n_samples):
            condition = y_train[i] * (np.dot(X_train[i], w) + b) >= 1
            if condition:
                # Correctly classified: only update for regularization
                grad_w = 2 * w
                grad_b = 0
            else:
                # Misclassified: update for both regularization and classification
                grad_w = 2 * w - np.dot(X_train[i], y_train[i])
                grad_b = -y_train[i]

            # Update weights and bias
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
    return w, b


def predict_linear_svm_bi(X, w, b):
    """
    Predict binary labels using a linear SVM.

    Parameters:
        X (numpy.ndarray): Testing feature matrix of shape (n_samples, n_features).
        w (numpy.ndarray): Weight vector of shape (n_features,).
        b (float): Bias term.

    Returns:
        numpy.ndarray: Predicted binary labels of shape (n_samples,).
    """
    return np.sign(np.dot(X, w) + b)




# Step 9.3: Gaussian SVM
def rbf_kernel_bi(x1, x2, gamma):
    """
    Compute the Gaussian (RBF) kernel between two samples.

    Parameters:
        x1 (numpy.ndarray): First sample of shape (n_features,).
        x2 (numpy.ndarray): Second sample of shape (n_features,).
        gamma (float): Kernel coefficient.

    Returns:
        float: Kernel value.
    """
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

def train_gaussian_svm_bi(X_train, y_train, gamma=0.5, C=1.0, epochs=1000):
    """
    Train a Gaussian Kernel SVM for binary classification.

    Parameters:
        X_train (numpy.ndarray): Training feature matrix of shape (n_samples, n_features).
        y_train (numpy.ndarray): Binary labels of shape (n_samples,) with values -1 and 1.
        gamma (float): Kernel coefficient for the RBF kernel.
        C (float): Regularization parameter.
        epochs (int): Number of iterations.

    Returns:
        alpha (numpy.ndarray): Lagrange multipliers of shape (n_samples,).
        y_train (numpy.ndarray): Original binary labels (required for prediction).
        X_train (numpy.ndarray): Training data (required for prediction).
        gamma (float): Kernel coefficient (required for prediction).
    """
    n_samples = X_train.shape[0]
    alpha = np.zeros(n_samples)
    K = np.array([[rbf_kernel_bi(x1, x2, gamma) for x2 in X_train] for x1 in X_train])

    for epoch in range(epochs):
        for i in range(n_samples):
            gradient = 1 - np.sum(alpha * y_train * K[:, i])
            alpha[i] += 0.001 * gradient
            alpha[i] = np.clip(alpha[i], 0, C)

    return alpha, y_train, X_train, gamma


def predict_gaussian_svm_bi(X_train, X_test, alpha, y_train, gamma):
    """
    Predict binary labels using a Gaussian Kernel SVM.

    Parameters:
        X_train (numpy.ndarray): Training feature matrix of shape (n_train_samples, n_features).
        X_test (numpy.ndarray): Testing feature matrix of shape (n_test_samples, n_features).
        alpha (numpy.ndarray): Lagrange multipliers of shape (n_train_samples,).
        y_train (numpy.ndarray): Original binary labels of shape (n_train_samples,).
        gamma (float): Kernel coefficient for the RBF kernel.

    Returns:
        y_pred (numpy.ndarray): Predicted labels of shape (n_test_samples,) with values -1 and 1.
    """
    n_test_samples = X_test.shape[0]
    scores = np.zeros(n_test_samples)

    for i, x_test in enumerate(X_test):
        scores[i] = np.sum(alpha * y_train * np.array([rbf_kernel_bi(x_train, x_test, gamma) for x_train in X_train]))

    return np.sign(scores)

# Step 9.4: AdaBoost
def train_adaboost_bi(X, y, n_estimators):
    n_samples = len(X)
    weights = np.ones(n_samples) / n_samples
    classifiers = []
    alphas = []

    for _ in range(n_estimators):
        # Weak classifier: use a single feature threshold
        best_feature, best_threshold, best_polarity, best_error = 0, 0, 1, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = polarity * np.sign(X[:, feature] - threshold)
                    error = np.sum(weights[predictions != y])
                    if error < best_error:
                        best_feature, best_threshold, best_polarity = feature, threshold, polarity
                        best_error = error
        alpha = 0.5 * np.log((1 - best_error) / max(best_error, 1e-10))
        predictions = best_polarity * np.sign(X[:, best_feature] - best_threshold)
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)
        classifiers.append((best_feature, best_threshold, best_polarity))
        alphas.append(alpha)
    return classifiers, alphas

def predict_adaboost_bi(X, classifiers, alphas):
    final_predictions = np.zeros(len(X))
    for alpha, (feature, threshold, polarity) in zip(alphas, classifiers):
        predictions = polarity * np.sign(X[:, feature] - threshold)
        final_predictions += alpha * predictions
    return np.sign(final_predictions)
#evaluate binary classifaication
def evaluate_binary_classification(y_true, y_pred):
    """
    Evaluate binary classification results using numpy.

    Parameters:
        y_true (numpy.ndarray): True labels (-1 or 1).
        y_pred (numpy.ndarray): Predicted labels (-1 or 1).

    Returns:
        accuracy (float): Classification accuracy.
        precision (float): Precision score.
        recall (float): Recall score.
        f1 (float): F1 score.
    """

    # Calculate confusion matrix elements
    true_positive = np.sum((y_true == 1) & (y_pred == 1))  # Correctly predicted positive class
    true_negative = np.sum((y_true != 1) & (y_pred != 1))  # Correctly predicted negative class
    false_positive = np.sum((y_true != 1) & (y_pred == 1))  # Negative class incorrectly predicted as positive
    false_negative = np.sum((y_true == 1) & (y_pred != 1))  # Positive class incorrectly predicted as negative

    # Calculate accuracy, precision, recall, and F1 score
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 score = 2 * (precision * recall) / (precision + recall)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Output each metric
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

    # Output the confusion matrix elements
    print(f"True Positive: {true_positive}")
    print(f"True Negative: {true_negative}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    
    return accuracy, precision, recall, f1


# Main program
def main():
    file_path = "seeds_dataset.txt" 
    X, y = load_data(file_path)
    X_normalized = normalize_features(X)
    X_train, X_test, y_train, y_test = _train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    
    # 1. K-Means++ Clustering
    print("K-Means++ Clustering:")
    kpp = Kpp()
    k = 3
    centroids, labels = kpp.kmeans(X_normalized, k)
    print("Final Centroids:\n", centroids)
    # Assuming data X, centroids, and labels are already given
    kpp.visualize_kmeans(X_normalized, centroids, labels, title="K-Means++ Clustering")

    # 2. Soft K-Means Clustering
    print("\nSoft K-Means Clustering:")
    soft_kpp = soft_Kpp()
    centroids, probabilities = soft_kpp.soft_kmeans(X_normalized, k, beta=2.0)
    print("Final Centroids:\n", centroids)
    print("First 5 Soft Assignments:\n", probabilities[:5])
    # Assuming data X, centroids, and soft assignment probabilities are already given.

    soft_kpp.visualize_soft_kmeans(X_normalized, centroids, probabilities, title="Soft K-Means Clustering")


    3.# Apply PCA
    for n_components in [2, 3]:
        pca = PCA(n_components=n_components)
        pca.fit(X_normalized)
        X_transformed = pca.transform(X_normalized)

        # Plot the principal components
        pca.plot_principal_components(X_transformed, y, n_components)
    
    4.# Apply nonlinear autoencoder
    for latent_dim in [2, 3]:
        print(f"Training Nonlinear AutoEncoder with latent dimension = {latent_dim}")
        autoencoder = NonlinearAutoEncoder(input_dim=X_normalized.shape[1], latent_dim=latent_dim)
        autoencoder.train(X_normalized, epochs=100, batch_size=16)
        X_encoded = autoencoder.encode(X_normalized)
        autoencoder.plot_encoded_data(X_encoded, y, latent_dim)


    5.# Apply clustering
    # Function for clustering and visualization
    clustering=Clustering()
    def apply_clustering(X_data, y_true, title_suffix="Original Data"):
        centroids_kmeans, labels_kmeans = kpp.kmeans(X_data, k)
        kpp.visualize_kmeans(X_data, centroids_kmeans, labels_kmeans, title=f"K-Means++ on {title_suffix}")
        _, probabilities_soft_kmeans = soft_kpp.soft_kmeans(X_data, k, beta=beta)
        labels_soft_kmeans = np.argmax(probabilities_soft_kmeans, axis=1)
        soft_kpp.visualize_soft_kmeans(X_data, centroids_kmeans, probabilities_soft_kmeans, title=f"Soft K-Means on {title_suffix}")
        # Compare metrics
        mapped_kmeans = clustering.map_cluster_to_labels(labels_kmeans, y_true)
        metrics_kmeans = clustering.calculate_metrics(y_true, mapped_kmeans)
        print(f"K-Means++ Metrics on {title_suffix}:", metrics_kmeans)
        mapped_soft_kmeans = clustering.map_cluster_to_labels(labels_soft_kmeans, y_true)
        metrics_soft_kmeans = clustering.calculate_metrics(y_true, mapped_soft_kmeans)
        print(f"Soft K-Means Metrics on {title_suffix}:", metrics_soft_kmeans)
        return metrics_kmeans, metrics_soft_kmeans    
    k = 3
    beta = 2.0
    dimensions = [2, 3]
    # Clustering on original data
    print("Clustering on Original Data:")
    metrics_kmeans, metrics_soft_kmeans = apply_clustering(X_normalized, y, "Original Data")

    for dim in dimensions:
        # PCA-based clustering
        print(f"\nClustering on PCA Reduced Data ({dim}D):")
        pca = PCA(n_components=n_components)
        pca.fit(X_normalized)
        X_pca = pca.transform(X_normalized)
        metrics_kmeans_pca, metrics_soft_kmeans_pca = apply_clustering(X_pca, y, f"PCA {dim}D")

        # Autoencoder-based clustering
        print(f"\nClustering on Autoencoder Reduced Data ({dim}D):")
        autoencoder = NonlinearAutoEncoder(input_dim=X_normalized.shape[1], latent_dim=dim)
        autoencoder.train(X_normalized, epochs=50, batch_size=16)
        X_encoded = autoencoder.encode(X_normalized)
        metrics_kmeans_autoencoder, metrics_soft_kmeans_autoencoder = apply_clustering(X_encoded, y, f"Autoencoder {dim}D")

    6.# Apply MLP
    print("\nApplying MLP:")
    mlp=MLP()
    y_one_hot = mlp.preprocess_labels(y, num_classes=3)
    y_train_one_hot, y_test_one_hot = mlp.preprocess_labels(y_train, num_classes=3), mlp.preprocess_labels(y_test, num_classes=3)

    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = 3
    epochs = 500
    learning_rate = 0.1
    weights = mlp.train_mlp(X_train, y_train_one_hot, input_dim, hidden_dim, output_dim, epochs, learning_rate)

    y_pred_mlp_mul = mlp.predict(X_test, weights)
    mlp_accuracy = np.mean(y_pred_mlp_mul == y_test)
    print(f"MLP Accuracy: {mlp_accuracy * 100:.2f}%")

    # Compare MLP with clustering results
    print("\nComparing MLP with Clustering Results:")
    print(f"MLP Accuracy: {mlp_accuracy:.4f}")

    print("Comparing MLP with K-Means++ and Soft K-Means on Original Data:")
    print(f"K-Means++ Accuracy: {metrics_kmeans['Accuracy']:.4f}")
    print(f"Soft K-Means Accuracy: {metrics_soft_kmeans['Accuracy']:.4f}")

    for dim in dimensions:
        print(f"\nComparing MLP with K-Means++ and Soft K-Means on PCA {dim}D Data:")
        print(f"K-Means++ Accuracy on PCA {dim}D: {metrics_kmeans_pca['Accuracy']:.4f}")
        print(f"Soft K-Means Accuracy on PCA {dim}D: {metrics_soft_kmeans_pca['Accuracy']:.4f}")

        print(f"\nComparing MLP with K-Means++ and Soft K-Means on Autoencoder {dim}D Data:")
        print(f"K-Means++ Accuracy on Autoencoder {dim}D: {metrics_kmeans_autoencoder['Accuracy']:.4f}")
        print(f"Soft K-Means Accuracy on Autoencoder {dim}D: {metrics_soft_kmeans_autoencoder['Accuracy']:.4f}")


    7.# Apply svm 
    # Linear SVM Multi-class
    print("Training Linear SVM for Multi-class...")
    svm=SVM()
    linear_classifiers = svm.train_linear_svm_multi(X_train, y_train)
    y_pred_linear = svm.predict_linear_svm_multi(X_test, linear_classifiers)
    evaluate_multiclass_classification(y_test, y_pred_linear)

    # Gaussian SVM Multi-class
    print("Training Gaussian Kernel SVM for Multi-class...")
    gaussian_classifiers = svm.train_gaussian_svm_multi(X_train, y_train)
    y_pred_gaussian = svm.predict_gaussian_svm_multi(X_train, X_test, gaussian_classifiers)
    evaluate_multiclass_classification(y_test, y_pred_gaussian)

    
    8.# Apply adaboost
    # Train AdaBoost Multi-class
    print("Training AdaBoost for Multi-class...")
    adaboost=Adaboost()
    adaboost_models = adaboost.train_adaboost_multi(X_train, y_train, n_estimators=50)

    # Predict and evaluate
    y_pred = adaboost.predict_adaboost_multi(X_test, adaboost_models)
    accuracy = np.mean(y_pred == y_test)
    print(f"AdaBoost Multi-class Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(np.histogram2d(y_test, y_pred, bins=[np.arange(1, 5), np.arange(1, 5)])[0])

    9.#binary classification
    X_train_bi,y_train_bi=prepare_binary_data(X_train,y_train)
    X_test_bi,y_test_bi=prepare_binary_data(X_test,y_test)
    # MLP
    print("Training MLP...")
    y_test_bi_mlp = np.where(y_test_bi == 1, 1, 0) 
    y_train_bi_mlp = np.where(y_train_bi == 1, 1, 0) 
    weights_mlp = train_mlp_bi(X_train_bi, y_train_bi_mlp, input_dim=X_train_bi.shape[1], hidden_dim=32, learning_rate=0.001, epochs=1000)
    y_pred_mlp_bi = predict_mlp_bi(X_test_bi, weights_mlp)
    evaluate_binary_classification(y_test_bi_mlp, y_pred_mlp_bi.flatten())

    

    # Linear SVM
    print("Training Linear SVM...")
    w, b = train_linear_svm_bi(X_train_bi, y_train_bi, learning_rate=0.0001, epochs=500, C=0.1)
    y_pred_svm_bi = predict_linear_svm_bi(X_test_bi, w, b)
    evaluate_binary_classification(y_test_bi, y_pred_svm_bi)

    # Gaussian SVM
    print("Training Gaussian Kernel SVM...")
    alpha, y_train_bi_, X_train_bi_, gamma = train_gaussian_svm_bi(X_train_bi, y_train_bi, gamma=0.5, C=1.0, epochs=1000)
    y_pred_gaussian = predict_gaussian_svm_bi(X_train_bi_, X_test_bi, alpha, y_train_bi_, gamma)
    evaluate_binary_classification(y_test_bi, y_pred_gaussian)

    # AdaBoost
    print("Training AdaBoost...")
    classifiers, alphas = train_adaboost_bi(X_train_bi, y_train_bi, n_estimators=50)
    y_pred_adaboost_bi = predict_adaboost_bi(X_test_bi, classifiers, alphas)
    evaluate_binary_classification(y_test_bi, y_pred_adaboost_bi)

if __name__ == "__main__":
    main()

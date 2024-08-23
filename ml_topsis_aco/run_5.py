import wandb
import numpy as np
import random
from scipy.stats import beta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, hamming_loss, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.dataset import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize W&B project
wandb.init(project="ML-TOPSIS-ACO", name="bibtex-ml-classification")

# Load the BibTeX dataset (train and test variants)
train_data = load_dataset("bibtex", "train")
test_data = load_dataset("bibtex", "test")

# Split features and labels for both train and test data
X_train, y_train = train_data[0], train_data[1]
X_test, y_test = test_data[0], test_data[1]

# Ensure that the feature matrices and label vectors are numpy arrays
X_train = X_train.toarray() if not isinstance(X_train, np.ndarray) else X_train
y_train = y_train.toarray() if not isinstance(y_train, np.ndarray) else y_train
X_test = X_test.toarray() if not isinstance(X_test, np.ndarray) else X_test
y_test = y_test.toarray() if not isinstance(y_test, np.ndarray) else y_test

# Combine train and test data for unified processing
X = np.vstack((X_train, X_test))
y = np.vstack((y_train, y_test))

# Split the combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Function to perform MTOPSIS feature ranking
def mtopis_feature_ranking(X, y, topf):
    """
    Rank features based on the Modified TOPSIS method.

    Parameters:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Label matrix.
        topf (int): Number of top features to select.

    Returns:
        numpy.ndarray: Indices of selected features.
    """
    # 1. Train a Ridge regression model to get feature-label correlations
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    feature_label_corr = ridge.coef_

    # 2. Compute feature-feature correlations using Pearson correlation coefficient
    feature_feature_corr = np.corrcoef(X, rowvar=False)

    # 3. Compute entropy-based weights for each label
    entropy = -np.sum(y * np.log(y + 1e-9), axis=0)
    weights = entropy / np.sum(entropy)

    # Ensure weights are of the correct shape (match number of features in X)
    weights_full = np.zeros(X.shape[1])
    weights_full[:len(weights)] = weights
    weights_normalized = normalize(weights_full.reshape(1, -1), norm='l2').flatten()

    # 4. Compute the weighted decision matrix
    weighted_decision_matrix = np.dot(X, weights_normalized)

    # 5. Calculate separation measures from the ideal best and worst
    ideal_best = np.max(weighted_decision_matrix)
    ideal_worst = np.min(weighted_decision_matrix)

    separation_best = np.sqrt(np.sum((weighted_decision_matrix - ideal_best) ** 2))
    separation_worst = np.sqrt(np.sum((weighted_decision_matrix - ideal_worst) ** 2))

    # 6. Compute relative closeness (RC) to the ideal solution
    relative_closeness = separation_worst / (separation_best + separation_worst)
    print(f"relative_closeness: {relative_closeness}")
    
    # 7. Rank the features based on RC and select the topf features
    feature_ranking = np.argsort(-relative_closeness)
    print(f"feature_ranking: {feature_ranking}")
    selected_features = feature_ranking[:topf]

    return selected_features

# Function to perform Modified ACO feature reranking
def modified_aco_feature_reranking(X, y, selected_features, num_iterations=100, num_ants=10, decay_rate=0.3):
    """
    Re-rank selected features using a Modified Ant Colony Optimization (ACO) algorithm.

    Parameters:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Label matrix.
        selected_features (numpy.ndarray): Initially selected feature indices.
        num_iterations (int): Number of ACO iterations.
        num_ants (int): Number of ants in each iteration.
        decay_rate (float): Pheromone decay rate.

    Returns:
        tuple: Optimal features after re-ranking and the final pheromone values.
    """
    # 1. Initialize Pheromones based on Jaccard similarity
    pheromones = np.zeros(len(selected_features))

    for idx, feature_idx in enumerate(selected_features):
        jaccard_similarity = np.sum(y[:, feature_idx]) / (np.sum(X[:, feature_idx]) + np.sum(y[:, feature_idx]) - np.sum(X[:, feature_idx] * y[:, feature_idx]))
        pheromones[idx] = jaccard_similarity

    pheromones = normalize(pheromones.reshape(1, -1), norm='l1').flatten()

    # Adjust num_ants if it's larger than the number of selected features
    num_ants = min(num_ants, len(selected_features))

    # 2. Set ACO parameters and start iterations
    for iteration in range(num_iterations):
        for ant in range(num_ants):
            # 3. Randomly select features for the ants to explore
            selected_feature_indices = random.sample(range(len(selected_features)), num_ants)

            # 4. Move ant to the next state by applying a probability-based heuristic function (using Beta Distribution)
            for feature_idx in selected_feature_indices:
                beta_distribution = beta.rvs(2, 5, size=1)
                next_feature_idx = np.argmax(pheromones * beta_distribution)

                # 5. Update Pheromone value based on movement
                pheromones[feature_idx] = pheromones[feature_idx] * (1 - decay_rate) + pheromones[next_feature_idx]

    # 6. Rank the features based on Pheromone values and return the optimal features
    final_ranking = np.argsort(-pheromones)
    optimal_features = selected_features[final_ranking]

    return optimal_features, pheromones

# Apply MTOPSIS for initial feature selection
topf = max(100, int(0.40 * X_train.shape[1]))  # Select at least 10 features or 20% of total features, whichever is larger
print(f"topf: {topf}")

selected_features = mtopis_feature_ranking(X_train, y_train, topf)
print(f"selected_features: {selected_features}")
# Apply Modified-ACO for feature re-ranking
optimal_features, final_pheromones = modified_aco_feature_reranking(X_train, y_train, selected_features)

# Reduce the training and testing data based on optimal features
X_train_reduced = X_train[:, optimal_features]
X_test_reduced = X_test[:, optimal_features]

# Train KNeighborsClassifier on the reduced dataset
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_reduced, y_train)

# Predict on the test set
predictions = knn.predict(X_test_reduced)

# Evaluate the model using Accuracy and Hamming Loss
accuracy = accuracy_score(y_test, predictions)

average_precision = average_precision_score(y_test, predictions)
print(f"average_precision_score: {average_precision}")

hamming_loss_score = hamming_loss(y_test, predictions)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")
print(f"Hamming Loss: {hamming_loss_score:.4f}")
print(f"Average Precision score: {average_precision:.4f}")

# Log the results to W&B
wandb.log({
    "Accuracy": accuracy,
    "Hamming Loss": hamming_loss_score,
    "Number of Selected Features": len(optimal_features)
})

# Plot Feature Importance based on Pheromone Values
plt.figure(figsize=(10, 6))
plt.bar(range(len(optimal_features)), final_pheromones)
plt.title("Feature Importance based on Pheromone Values")
plt.xlabel("Feature Index")
plt.ylabel("Pheromone Value")
wandb.log({"Feature Importance": wandb.Image(plt)})

# Plot Accuracy vs Number of Selected Features
plt.figure(figsize=(10, 6))
plt.plot(range(10, 101, 10), [accuracy for _ in range(10, 101, 10)], marker='o')
plt.title("Accuracy vs Number of Selected Features")
plt.xlabel("Number of Selected Features")
plt.ylabel("Accuracy")
plt.xlim([0, 100])
plt.ylim([0, 1])
wandb.log({"Accuracy vs Features": wandb.Image(plt)})

# Plot Hamming Loss vs Number of Selected Features
plt.figure(figsize=(10, 6))
plt.plot(range(10, 101, 10), [hamming_loss_score for _ in range(10, 101, 10)], marker='o')
plt.title("Hamming Loss vs Number of Selected Features")
plt.xlabel("Number of Selected Features")
plt.ylabel("Hamming Loss")
plt.xlim([0, 100])
plt.ylim([0, 1])
wandb.log({"Hamming Loss vs Features": wandb.Image(plt)})

# End the WandB run
wandb.finish()

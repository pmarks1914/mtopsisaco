import wandb
import numpy as np
import random
from scipy.stats import beta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
from skmultilearn.dataset import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize W&B project
wandb.init(project="ML-TOPSIS-ACO", name="bibtex-ml-classification")

# Load the BibTeX dataset (both train and test variants)
train_data = load_dataset("bibtex", "train")
test_data = load_dataset("bibtex", "test")

X_train, y_train = train_data[0], train_data[1]
X_test, y_test = test_data[0], test_data[1]

# Ensure X and y are numpy arrays
if not isinstance(X_train, np.ndarray):
    X_train = X_train.toarray()
if not isinstance(y_train, np.ndarray):
    y_train = y_train.toarray()
if not isinstance(X_test, np.ndarray):
    X_test = X_test.toarray()
if not isinstance(y_test, np.ndarray):
    y_test = y_test.toarray()

# Combine train and test data for unified processing
X = np.vstack((X_train, X_test))
y = np.vstack((y_train, y_test))

# Split the combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def mtopis_feature_ranking(X, y, topf):
    # 1. Create decision matrix C
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    feature_label_corr = ridge.coef_
    feature_feature_corr = np.corrcoef(X, rowvar=False)

    # 2. Compute label weight matrix using entropy
    entropy = -np.sum(y * np.log(y + 1e-9), axis=0)
    weights = entropy / np.sum(entropy)

    # Ensure weights are of the correct shape (number of features in X)
    weights_full = np.zeros(X.shape[1])
    weights_full[:len(weights)] = weights
    weights_normalized = normalize(weights_full.reshape(1, -1), norm='l2').flatten()

    # 4. Compute weighted decision matrix
    weighted_decision_matrix = np.dot(X, weights_normalized)

    # 5. Calculate separation measures from PD+ and NB-
    ideal_best = np.max(weighted_decision_matrix)
    ideal_worst = np.min(weighted_decision_matrix)

    separation_best = np.sqrt(np.sum((weighted_decision_matrix - ideal_best) ** 2))
    separation_worst = np.sqrt(np.sum((weighted_decision_matrix - ideal_worst) ** 2))

    # 6. Compute relative closeness (RC)
    relative_closeness = separation_worst / (separation_best + separation_worst)

    # 7. Rank the features based on RC and choose the topf features
    feature_ranking = np.argsort(-relative_closeness)
    selected_features = feature_ranking[:topf]

    return selected_features

def modified_aco_feature_reranking(X, y, selected_features, num_iterations=100, num_ants=10, decay_rate=0.5):
    # 1. Initialize Pheromones with Jaccard similarity
    pheromones = np.zeros(len(selected_features))

    for idx, feature_idx in enumerate(selected_features):
        jaccard_similarity = np.sum(y[:, feature_idx]) / (np.sum(X[:, feature_idx]) + np.sum(y[:, feature_idx]) - np.sum(X[:, feature_idx] * y[:, feature_idx]))
        pheromones[idx] = jaccard_similarity

    pheromones = normalize(pheromones.reshape(1, -1), norm='l1').flatten()

    # Adjust num_ants if it's larger than the number of selected features
    num_ants = min(num_ants, len(selected_features))

    # 2. Set ACO parameters
    for iteration in range(num_iterations):
        for ant in range(num_ants):
            # 3. Generate unique integers selected randomly from 1 to the number of selected features
            selected_feature_indices = random.sample(range(len(selected_features)), num_ants)

            # 4. Place ant randomly on each feature
            for feature_idx in selected_feature_indices:
                # 5. Move ant to the next state by applying probability-based heuristic function (using Beta Distribution)
                beta_distribution = beta.rvs(2, 5, size=1)
                next_feature_idx = np.argmax(pheromones * beta_distribution)

                # 6. Update Pheromone value
                pheromones[feature_idx] = pheromones[feature_idx] * (1 - decay_rate) + pheromones[next_feature_idx]

    # 7. Rank the features based on Pheromone value
    final_ranking = np.argsort(-pheromones)
    optimal_features = selected_features[final_ranking]

    return optimal_features, pheromones

# Apply MTOPSIS for initial feature selection
# topf = int(0.2 * X_train.shape[1])  # Select top 20% features
topf = max(10, int(0.2 * X_train.shape[1]))  # Select at least 10 features or 20% of total features, whichever is larger
selected_features = mtopis_feature_ranking(X_train, y_train, topf)

# Apply Modified-ACO for feature re-ranking
optimal_features, final_pheromones = modified_aco_feature_reranking(X_train, y_train, selected_features)

# Reduce the training and testing data based on optimal features
X_train_reduced = X_train[:, optimal_features]
X_test_reduced = X_test[:, optimal_features]

# Train MLKNN on the reduced dataset
mlknn = MLkNN(k=3)
mlknn.fit(X_train_reduced, y_train)

# Predict on test set
predictions = mlknn.predict(X_test_reduced)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions.toarray())
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

# Log the result to W&B
wandb.log({"Accuracy": accuracy})

# Log feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(optimal_features)), final_pheromones)
plt.title("Feature Importance based on Pheromone Values")
plt.xlabel("Feature Index")
plt.ylabel("Pheromone Value")
wandb.log({"Feature Importance": plt})

# End the WandB run
wandb.finish()
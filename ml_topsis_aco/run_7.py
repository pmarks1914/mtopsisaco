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
from tqdm import tqdm

# Initialize W&B project
wandb.init(project="ML-TOPSIS-ACO", name="bibtex-ml-classification")

print("Loading dataset...")
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

print("Preprocessing data...")
# Combine train and test data for unified processing
X = np.vstack((X_train, X_test))
y = np.vstack((y_train, y_test))

# Split the combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

def mtopis_feature_ranking(X, y, topf):
    print("Performing MTOPSIS feature ranking...")
    # 1. Train a Ridge regression model to get feature-label correlations
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    feature_label_corr = np.abs(ridge.coef_).mean(axis=0)

    # 2. Compute feature-feature correlations using Pearson correlation coefficient
    feature_feature_corr = np.abs(np.corrcoef(X.T))

    # 3. Compute entropy-based weights for each feature
    entropy = -np.sum((X / X.sum(axis=0)) * np.log2(X / X.sum(axis=0) + 1e-10), axis=0)
    weights = entropy / np.sum(entropy)

    # 4. Compute the weighted decision matrix
    weighted_decision_matrix = feature_label_corr * weights

    # 5. Calculate separation measures from the ideal best and worst
    ideal_best = np.max(weighted_decision_matrix)
    ideal_worst = np.min(weighted_decision_matrix)

    separation_best = np.sqrt(np.sum((weighted_decision_matrix - ideal_best) ** 2, axis=0))
    separation_worst = np.sqrt(np.sum((weighted_decision_matrix - ideal_worst) ** 2, axis=0))

    # 6. Compute relative closeness (RC) to the ideal solution
    relative_closeness = separation_worst / (separation_best + separation_worst)
    
    # 7. Rank the features based on RC and select the topf features
    feature_ranking = np.argsort(-relative_closeness)
    selected_features = feature_ranking[:topf]

    return selected_features

def modified_aco_feature_reranking(X, y, selected_features, num_iterations=50, num_ants=5, decay_rate=0.3):
    print("Performing Modified ACO feature reranking...")
    pheromones = np.ones(len(selected_features))
    best_features = selected_features.copy()
    best_score = 0

    for iteration in tqdm(range(num_iterations), desc="ACO Iterations"):
        for ant in range(num_ants):
            # Select a subset of features
            subset_size = random.randint(max(1, len(selected_features) // 2), len(selected_features))
            ant_features = np.random.choice(selected_features, subset_size, p=pheromones/sum(pheromones), replace=False)
            
            # Evaluate the subset
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X[:, ant_features], y)
            score = accuracy_score(y, knn.predict(X[:, ant_features]))
            
            # Update best solution if necessary
            if score > best_score:
                best_score = score
                best_features = ant_features
            
            # Update pheromones
            pheromones[np.isin(selected_features, ant_features)] += score
        
        # Apply decay
        pheromones *= (1 - decay_rate)

    return best_features, pheromones

print("Starting feature selection...")
# Apply MTOPSIS for initial feature selection
topf = max(100, int(0.20 * X_train.shape[1]))  # Reduced from 0.40 to 0.20
selected_features = mtopis_feature_ranking(X_train, y_train, topf)

# Apply Modified-ACO for feature re-ranking
optimal_features, final_pheromones = modified_aco_feature_reranking(X_train, y_train, selected_features)

print("Feature selection complete. Training model...")
# Reduce the training and testing data based on optimal features
X_train_reduced = X_train[:, optimal_features]
X_test_reduced = X_test[:, optimal_features]

# Train KNeighborsClassifier on the reduced dataset
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_reduced, y_train)

print("Making predictions...")
# Predict on the test set
predictions = knn.predict(X_test_reduced)

print("Evaluating model...")
# Evaluate the model using Accuracy and Hamming Loss
accuracy = accuracy_score(y_test, predictions)
average_precision = average_precision_score(y_test, predictions, average='samples')
hamming_loss_score = hamming_loss(y_test, predictions)

print(f"Accuracy of the model: {accuracy * 100:.2f}%")
print(f"Hamming Loss: {hamming_loss_score:.4f}")
print(f"Average Precision score: {average_precision:.4f}")
print(f"Number of selected features: {len(optimal_features)}")

# Log the results to W&B
wandb.log({
    "Accuracy": accuracy,
    "Hamming Loss": hamming_loss_score,
    "Average Precision": average_precision,
    "Number of Selected Features": len(optimal_features)
})

# Plot Feature Importance based on Pheromone Values
plt.figure(figsize=(10, 6))
plt.bar(range(len(optimal_features)), final_pheromones)
plt.title("Feature Importance based on Pheromone Values")
plt.xlabel("Feature Index")
plt.ylabel("Pheromone Value")
wandb.log({"Feature Importance": wandb.Image(plt)})

# End the WandB run
wandb.finish()
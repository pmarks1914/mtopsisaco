import wandb
import numpy as np
import random
from scipy.stats import beta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, hamming_loss, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.dataset import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from skmultilearn.adapt import MLkNN

# Initialize W&B project
wandb.init(project="ML-TOPSIS-ACO-SMALL-RVM", name="bibtex-ml-classification")

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

def mtopis_feature_ranking(X, y, topf):
    """Perform MTOPSIS feature ranking.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label matrix.
        topf (int): Number of top features to select.
        
    Returns:
        np.ndarray: Indices of selected features.
    """
    print("Performing MTOPSIS feature ranking...")
    print(f"Input shape: X={X.shape}, y={y.shape}")
    
    # 1. Train a Ridge regression model to get feature-label correlations
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    feature_label_corr = np.abs(ridge.coef_).mean(axis=0)
    print(f"Feature-label correlation shape: {feature_label_corr.shape}")
    print(f"Feature-label correlation range: [{feature_label_corr.min()}, {feature_label_corr.max()}]")

    # 2. Compute feature-feature correlations using Pearson correlation coefficient
    feature_feature_corr = np.abs(np.corrcoef(X.T))
    print(f"Feature-feature correlation shape: {feature_feature_corr.shape}")

    # 3. Compute entropy-based weights for each feature
    epsilon = 1e-10
    X_norm = X / (X.sum(axis=0) + epsilon)
    entropy = -np.sum(X_norm * np.log2(X_norm + epsilon), axis=0)
    weights = entropy / (np.sum(entropy) + epsilon)
    print(f"Weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min()}, {weights.max()}]")

    # 4. Compute the weighted decision matrix
    weighted_decision_matrix = feature_label_corr * weights
    print(f"Weighted decision matrix shape: {weighted_decision_matrix.shape}")
    print(f"Weighted decision matrix range: [{weighted_decision_matrix.min()}, {weighted_decision_matrix.max()}]")

    # 5. Calculate separation measures from the ideal best and worst
    ideal_best = np.max(weighted_decision_matrix)
    ideal_worst = np.min(weighted_decision_matrix)

    separation_best = np.sqrt((weighted_decision_matrix - ideal_best) ** 2)
    separation_worst = np.sqrt((weighted_decision_matrix - ideal_worst) ** 2)
    print(f"Separation measures shape: best {separation_best.shape}, worst {separation_worst.shape}")

    # 6. Compute relative closeness (RC) to the ideal solution
    relative_closeness = separation_worst / (separation_best + separation_worst + epsilon)
    print(f"Relative closeness shape: {relative_closeness.shape}")
    print(f"Relative closeness range: [{relative_closeness.min()}, {relative_closeness.max()}]")
    
    # 7. Rank the features based on RC and select the topf features
    feature_ranking = np.argsort(-relative_closeness)
    selected_features = feature_ranking[:topf]

    print(f"Number of features selected by MTOPSIS: {len(selected_features)}")
    print(f"Top 5 feature indices: {selected_features[:5]}")
    
    if relative_closeness.ndim == 0:
        print("Cannot print relative closeness values for top features as it's a scalar.")
    else:
        print(f"Relative closeness values for top 5 features: {relative_closeness[selected_features[:5]]}")
    
    return selected_features

def modified_aco_feature_reranking(X, y, selected_features, num_iterations=5, num_ants=10, decay_rate=0.3):
    """Perform Modified ACO feature reranking.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label matrix.
        selected_features (np.ndarray): Initial selected features.
        num_iterations (int): Number of iterations.
        num_ants (int): Number of ants.
        decay_rate (float): Decay rate of pheromones.
        
    Returns:
        tuple: Optimized features and final pheromones.
    """
    print("Performing Modified ACO feature reranking...")
    print(f"Number of selected features: {len(selected_features)}")
    
    if len(selected_features) < 10:
        print("Too few features selected. Returning original features.")
        return selected_features, np.ones(len(selected_features))

    pheromones = np.ones(len(selected_features))
    best_features = selected_features.copy()
    best_score = 0

    for iteration in tqdm(range(num_iterations), desc="ACO Iterations"):
        for ant in range(num_ants):
            # Select a subset of features
            subset_size = random.randint(max(2, len(selected_features) // 10), max(3, len(selected_features) // 2))
            ant_features = np.random.choice(selected_features, subset_size, p=pheromones/sum(pheromones), replace=False)
            
            # # Evaluate the subset using MLkNN classifier
            # knn = MLkNN(k=3)
            # knn.fit(X[:, ant_features], y)
            # score = accuracy_score(y, knn.predict(X[:, ant_features]))
            
            # Use KNeighborsClassifier instead of MLkNN
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X[:, ant_features], y)
            score = accuracy_score(y, knn.predict(X[:, ant_features]))
            
            # Update best solution if necessary
            if score > best_score:
                best_score = score
                best_features = ant_features
            
            # Update pheromones based on the score
            pheromones[np.isin(selected_features, ant_features)] += score
        
        # Apply decay to pheromones
        pheromones *= (1 - decay_rate)

    print(f"Number of features selected by ACO: {len(best_features)}")
    print(f"Top 5 feature indices: {best_features[:5]}")
    
    # After the ACO iterations, sort the pheromones
    final_pheromones = pheromones[np.argsort(-pheromones)][:len(best_features)]
    
    return best_features, final_pheromones

print("Starting feature selection...")
# Apply MTOPSIS for initial feature selection
topf = max(100, int(0.30 * X_train.shape[1]))  # Increased from 0.20 to 0.30
print(f"Selecting top {topf} features")

print("Feature statistics:")
print(f"Mean of X_train: {np.mean(X_train)}")
print(f"Std of X_train: {np.std(X_train)}")
print(f"Min of X_train: {np.min(X_train)}")
print(f"Max of X_train: {np.max(X_train)}")
print(f"Number of non-zero elements in X_train: {np.count_nonzero(X_train)}")
print(f"Sparsity of X_train: {1 - np.count_nonzero(X_train) / X_train.size}")
print(f"Mean of y_train: {np.mean(y_train)}")
print(f"Std of y_train: {np.std(y_train)}")
print(f"Min of y_train: {np.min(y_train)}")
print(f"Max of y_train: {np.max(y_train)}")

# MTOPSIS feature ranking
selected_features = mtopis_feature_ranking(X_train, y_train, topf)

# Apply Modified-ACO for feature re-ranking
optimal_features, final_pheromones = modified_aco_feature_reranking(X_train, y_train, selected_features)

print("Feature selection complete. Training model...")
# Reduce the training and testing data based on optimal features
X_train_reduced = X_train[:, optimal_features]
X_test_reduced = X_test[:, optimal_features]

# Train MLkNN on the reduced dataset
knn = MLkNN(k=9)

# Add feature scaling before training the model:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# Fit the model on scaled training data
knn.fit(X_train_scaled, y_train)

print("Making predictions...")
# Predict on the test set
predictions = knn.predict(X_test_scaled)

# Alternative classifier example:
# rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
# rf.fit(X_train_scaled, y_train)
# predictions = rf.predict(X_test_scaled)

print("Evaluating the model...")

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, predictions)
hamming = hamming_loss(y_test, predictions)
average_precision = average_precision_score(y_test, predictions.toarray())

print(f"Model accuracy: {accuracy}")
print(f"Model hamming loss: {hamming}")
print(f"Model average precision: {average_precision}")

# Log metrics to W&B
wandb.log({
    "accuracy": accuracy,
    "hamming_loss": hamming,
    "average_precision": average_precision
})

print("Run complete.")

# Close the W&B run
wandb.finish()

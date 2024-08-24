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
from sklearn.preprocessing import StandardScaler


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
# Combine train and test data for unified processing
X = np.vstack((X_train, X_test))
y = np.vstack((y_train, y_test))

# Split the combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

def mtopis_feature_ranking(X, y, topf):
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

def modified_aco_feature_reranking(X, y, selected_features, num_iterations=5, num_ants=5, decay_rate=0.3):
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

    print(f"Number of features selected by ACO: {len(best_features)}")
    print(f"Top 5 feature indices: {best_features[:5]}")
    
    # After the ACO iterations
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

selected_features = mtopis_feature_ranking(X_train, y_train, topf)

# Apply Modified-ACO for feature re-ranking
optimal_features, final_pheromones = modified_aco_feature_reranking(X_train, y_train, selected_features)

print("Feature selection complete. Training model...")
# Reduce the training and testing data based on optimal features
X_train_reduced = X_train[:, optimal_features]
X_test_reduced = X_test[:, optimal_features]

# Train KNeighborsClassifier on the reduced dataset
knn = KNeighborsClassifier(n_neighbors=3)

# Add feature scaling before training the model:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# knn.fit(X_train_reduced, y_train)

print("Making predictions...")
# Predict on the test set
# predictions = knn.predict(X_test_reduced)

knn.fit(X_train_scaled, y_train)
predictions = knn.predict(X_test_scaled)

"""
Try a different classifier, such as Random Forest, which might handle the multi-label classification better:
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf.fit(X_train_scaled, y_train)
predictions = rf.predict(X_test_scaled)

"""
print("Evaluating model...")
# Evaluate the model using Accuracy and Hamming Loss
accuracy = accuracy_score(y_test, predictions)
average_precision = average_precision_score(y_test, predictions, average='samples')
hamming_loss_score = hamming_loss(y_test, predictions)

print(f"Accuracy of the model: {accuracy * 100:.2f}%")
print(f"Hamming Loss: {hamming_loss_score * 100:.4f}%")
print(f"Average Precision score: {average_precision * 100:.4f}%")
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



"""
Logs

Loading dataset...
bibtex:train - exists, not redownloading
bibtex:test - exists, not redownloading
Preprocessing data...
Starting feature selection...
Selecting top 550 features
Feature statistics:
Mean of X_train: 0.03733175213486346
Std of X_train: 0.18957344860872402
Min of X_train: 0.0
Max of X_train: 1.0
Number of non-zero elements in X_train: 456141
Sparsity of X_train: 0.9626682478651365
Mean of y_train: 0.015052757419824316
Std of y_train: 0.12176276899727699
Min of y_train: 0
Max of y_train: 1
Performing MTOPSIS feature ranking...
Input shape: X=(6655, 1836), y=(6655, 159)
Feature-label correlation shape: (1836,)
Feature-label correlation range: [0.00252147898364939, 0.025933492569921068]
Feature-feature correlation shape: (1836, 1836)
Weights shape: (1836,)
Weights range: [0.00031578732185150665, 0.000959495107004753]
Weighted decision matrix shape: (1836,)
Weighted decision matrix range: [2.159880501667457e-06, 1.2597636946018777e-05]
Separation measures shape: best (1836,), worst (1836,)
Relative closeness shape: (1836,)
Relative closeness range: [0.0, 0.9999904194888545]
Number of features selected by MTOPSIS: 550
Top 5 feature indices: [1117 1823  951 1129 1428]
Relative closeness values for top 5 features: [0.99999042 0.94529446 0.94529446 0.89930608 0.87950366]
Performing Modified ACO feature reranking...
Number of selected features: 550
ACO Iterations: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [09:38<00:00, 115.62s/it]
Number of features selected by ACO: 238
Top 5 feature indices: [1175 1261 1751 1378   50]
Feature selection complete. Training model...
Making predictions...
Evaluating model...
Accuracy of the model: 9.05%
Hamming Loss: 0.0142
Average Precision score: 0.1718
Number of selected features: 238
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                    Accuracy ▁
wandb:           Average Precision ▁
wandb:                Hamming Loss ▁
wandb: Number of Selected Features ▁
wandb: 
wandb: Run summary:
wandb:                    Accuracy 0.09054
wandb:           Average Precision 0.17177
wandb:                Hamming Loss 0.01417
wandb: Number of Selected Features 238
wandb: 
wandb: 🚀 View run bibtex-ml-classification at: https://wandb.ai/sharhanalhassan-university-of-ghana/ML-TOPSIS-ACO-SMALL-RVM/runs/sczr8di0
wandb: ⭐️ View project at: https://wandb.ai/sharhanalhassan-university-of-ghana/ML-TOPSIS-ACO-SMALL-RVM
wandb: Synced 5 W&B file(s), 1 media file(s), 2 artifact file(s) and 0 other file(s)
"""
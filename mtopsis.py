import numpy as np
from sklearn.linear_model import Ridge
from tqdm import tqdm  # Import tqdm for the progress bar
import pandas as pd
import logging
from scipy.stats import beta
from sklearn.neighbors import KNeighborsClassifier
import wandb
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import MLkNN

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss
# Custom one_error function for multilabel classification
def one_error(y_true, y_pred):
    """
    Calculate the one-error score for multilabel classification.
    
    Parameters:
    y_true (array-like): True binary labels (n_samples, n_labels).
    y_pred (array-like): Predicted binary labels (n_samples, n_labels).
    
    Returns:
    float: One-error score.
    """
    # n_samples = y_true.shape[0]
    # one_error_count = 0
    # for i in range(n_samples):
    #     if not np.any(y_true[i] & y_pred[i]):
    #         one_error_count += 1
    # return one_error_count / n_samples
    return np.mean(np.sum(y_true & y_pred, axis=1) == 0)

# Custom ranking_loss function for multilabel classification
def ranking_loss(y_true, y_pred):
    """
    Calculate the ranking loss for multilabel classification.
    
    Parameters:
    y_true (array-like): True binary labels (n_samples, n_labels).
    y_pred (array-like): Predicted binary labels (n_samples, n_labels).
    
    Returns:
    float: Ranking loss.
    """
    n_samples = y_true.shape[0]
    ranking_loss_sum = 0.0
    
    for i in range(n_samples):
        true_labels = np.where(y_true[i] == 1)[0]
        predicted_labels = np.where(y_pred[i] == 1)[0]
        
        for tl in true_labels:
            for pl in predicted_labels:
                if tl != pl and y_true[i, pl] == 0:
                    ranking_loss_sum += 1
    
    n_possible_pairs = np.sum(y_true.sum(axis=1) * (y_true.shape[1] - y_true.sum(axis=1)))
    return ranking_loss_sum / n_possible_pairs if n_possible_pairs > 0 else 0

def compute_decision_matrix(X, Y, λ):
    """
    Compute decision matrix C with dimension M x N,
    combining FL and FF correlation calculated using Ridge regression for a multi-label dataset.
    
    Args:
        X (np.ndarray or pd.DataFrame): Input features
        Y (np.ndarray or pd.DataFrame): Target labels (multi-label)
        λ (float): Regularization parameter for Ridge regression
        
    Returns:
        np.ndarray: Decision matrix C
    """
    X = X.values if isinstance(X, pd.DataFrame) else X  # Convert X to NumPy array if it's a DataFrame
    Y = Y.values if isinstance(Y, pd.DataFrame) else Y  # Convert Y to NumPy array if it's a DataFrame
    
    n_features = X.shape[1]
    n_labels = Y.shape[1]
    
    # Calculate Feature-Label (FL) correlation using Ridge regression
    FL_correlation = np.zeros((n_features, n_labels))
    for j in tqdm(range(n_labels), desc='Calculating FL Correlation'):
        clf_fl = Ridge(alpha=λ)
        clf_fl.fit(X, Y[:, j])
        FL_correlation[:, j] = clf_fl.coef_
    
    # Calculate Feature-Feature (FF) correlation for each pair of features
    FF_correlation = np.zeros((n_features, n_features))
    
    for i in tqdm(range(n_features), desc='Calculating FF Correlation'):
        for j in range(i + 1, n_features):  # Avoid redundant calculations
            clf_ff = Ridge(alpha=λ)
            clf_ff.fit(X[:, [i]], X[:, j])
            FF_correlation[i, j] = FF_correlation[j, i] = clf_ff.coef_[0]
        FF_correlation[i, i] = 1  # Set self-correlation to 1
    
    # Combine FL and FF correlation to form decision matrix C
    C = np.concatenate((FL_correlation, FF_correlation), axis=1)
    return C


def calculate_weighted_matrix(C):
    """
    Calculate weighted matrix v for each label and feature.
    """
    v = np.sum(np.abs(C), axis=0) # Sum of absolute values of each column
    return v

def calculate_normalized_weighted_matrix(v):
    """
    Calculate normalized weighted matrix v'.
    """
    v_normalized = v / np.sum(v)
    return v_normalized

def calculate_pis_nis(C_normalized):
    """
    Calculate Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS).
    """
    A_plus = np.max(C_normalized, axis=0) # Max of each column
    A_minus = np.min(C_normalized, axis=0) # Min of each column
    return A_plus, A_minus

def calculate_separation(C_normalized, A_plus, A_minus):
    """
    Calculate separation from PIS (PD) and NIS (ND).
    """
    PD_star = np.square(C_normalized - A_plus).sum(axis=1) # Sum of squared differences from PIS
    ND_minus = np.square(C_normalized - A_minus).sum(axis=1) # Sum of squared differences from NIS
    return PD_star, ND_minus

def calculate_relative_closeness(PD_star, ND_minus):
    """
    Calculate Relative Closeness (RC).
    """
    D_star = np.sqrt(np.sum(PD_star)) # Euclidean distance from PIS
    D_minus = np.sqrt(np.sum(ND_minus)) # Euclidean distance from NIS
    RC = 0.5 * (ND_minus / D_minus) - 0.5 * (PD_star / D_star)
    return RC

def MTOPSIS(X, Y, λ):
    """
    Modified TOPSIS for feature ranking.
    """
    C = compute_decision_matrix(X, Y, λ)
    v = calculate_weighted_matrix(C)
    v_normalized = calculate_normalized_weighted_matrix(v)
    C_normalized = C * v_normalized
    A_plus, A_minus = calculate_pis_nis(C_normalized)
    PD_star, ND_minus = calculate_separation(C_normalized, A_plus, A_minus)
    RC = calculate_relative_closeness(PD_star, ND_minus)
    return RC
def select_top_features(X,RC, f):
    """
    Select the top f features based on their relative closeness values.
    """
    # Sort the RC array indices in descending order
    # Sort the RC array to get the indices of the features with the highest relative closeness
    top_indices = np.argsort(RC)[::-1][:f]

   

    return top_indices



# Configure logging
logging.basicConfig(level=logging.INFO)  # Set desired logging level

class ModifiedACO:
    def __init__(self, no_ants, no_cycles, t, q0, rho, beta_param):
        self.no_ants = no_ants
        self.no_cycles = no_cycles
        self.t = t
        self.q0 = q0
        self.rho = rho
        self.beta_param = beta_param
        self.logger = logging.getLogger(__name__)  # Create logger instance

    def cosine_similarity(self, X, y):
        X_norm = np.linalg.norm(X, axis=0, keepdims=True)
        y_norm = np.linalg.norm(y, axis=0, keepdims=True)

        if np.any(X_norm == 0) or np.any(y_norm == 0):
            similarity = np.zeros((X.shape[1], y.shape[1]))
        else:
            similarity = np.dot(X.T, y) / (X_norm.T @ y_norm)

        return similarity

    def calculate_correlations(self, X, y):
        corrf = np.corrcoef(X, rowvar=False)
        corrl = np.corrcoef(X, y, rowvar=False)[-1, :-1]
        return corrf, corrl

    def jaccard_similarity(self, X, y):
        intersection = np.dot(X.T, y)
        sum_X = X.sum(axis=0)
        sum_y = y.sum(axis=0)
        union = sum_X[:, None] + sum_y[None, :] - intersection
        jaccard = intersection / union
        return jaccard

    def initialize_pheromone(self, X, y):
        self.logger.info("Initializing pheromone...")
        jaccard = self.jaccard_similarity(X, y)
        pheromone = np.max(jaccard, axis=1)
        pheromone = (pheromone - np.min(pheromone)) / (np.max(pheromone) - np.min(pheromone))
        return pheromone

    def select_next_feature(self, pheromone, X, y, visited_features):
        unvisited_features = [j for j in range(X.shape[1]) if j not in visited_features]
        q = np.random.uniform()
        coeff_explore_exploit = (1 * (1 - len(visited_features) / X.shape[1]) ** 0.7)
        if np.random.beta(90, 10) > coeff_explore_exploit:
            next_feature = np.random.choice(unvisited_features)
        else:
            next_feature_values = [
                pheromone[j] * np.max(self.cosine_similarity(X[:, j].reshape(-1, 1), y)) ** self.beta_param
                for j in unvisited_features
            ]
            next_feature = unvisited_features[np.argmax(next_feature_values)]
        return next_feature

    def update_pheromone(self, pheromone, visited_features):
        for feature in visited_features:
            pheromone[feature] = (1 - self.rho) * pheromone[feature] + self.rho * np.sum(pheromone)
        return pheromone

    def normalize_pheromone(self, pheromone):
        return (pheromone - np.min(pheromone)) / (np.max(pheromone) - np.min(pheromone))

    def fit(self, X, y):
        self.logger.info("Fitting ACO model...")
        n_samples, n_features = X.shape
        pheromone = self.initialize_pheromone(X, y)
        sorted_indices = []

        self.logger.info("Starting ACO fitting.")

        for cycle in tqdm(range(self.no_cycles), desc="Cycles"):
            self.logger.debug(f"Starting cycle {cycle+1}/{self.no_cycles}")
            
            for ant in tqdm(range(self.no_ants), desc="Ants", leave=False):
                self.logger.debug(f"Ant {ant+1}/{self.no_ants} starting.")
                visited_features = []

                for _ in tqdm(range(n_features), desc="Features", leave=False):
                    next_feature = self.select_next_feature(pheromone, X, y, visited_features)
                    visited_features.append(next_feature)
                
                self.logger.debug(f"Ant {ant+1}/{self.no_ants} visited features: {visited_features}")
                pheromone = self.update_pheromone(pheromone, visited_features)
            
            pheromone = self.normalize_pheromone(pheromone)
            self.logger.debug(f"Cycle {cycle+1}/{self.no_cycles} complete.")

        sorted_indices = np.argsort(-pheromone)
        self.logger.info("ACO fitting complete.")
        return sorted_indices[:self.t]

        
def load_multilabel_data(dataset_name):
    # Load training dataset from skmultilearn
    X_train, y_train, feature_names_train, _ = load_dataset(dataset_name, 'train')
    
    # Convert sparse matrices to dense arrays
    X_train_dense = X_train.toarray()
    y_train_dense = y_train.toarray()
    
  
    # Load test dataset from skmultilearn
    X_test, y_test, feature_names_test, _ = load_dataset(dataset_name, 'test')
    
    # Convert sparse matrices to dense arrays
    X_test_dense = X_test.toarray()
    y_test_dense = y_test.toarray()
    
    # Convert label combinations to unique integers for y_test
    
    return X_train_dense, y_train_dense, X_test_dense, y_test_dense



# Assuming necessary imports for accuracy, one_error, ranking_loss, and the ModifiedACO class



# Assuming necessary imports for accuracy, one_error, ranking_loss, and the ModifiedACO class

def main(dataset_name):
    logging.info(f"Starting main function for dataset: {dataset_name}...")
    
    # Load dataset
    logging.info("Loading dataset...")
    X_train, y_train, X_test, y_test = load_multilabel_data(dataset_name)
     # Parameters
    λ = 10  # Example regularization parameter for MTOPSIS, adjust as needed
    no_ants = 25
    no_cycles = 40
    n_features = X_train.shape[1]//2 # Total number of features each ant has to visit
    q0 = 0.7
    rho = 0.1  # Pheromone decay rate
    beta_param = 1
    k_neighbors = 5  # Number of neighbors for KNN, adjust as needed
    f =X_train.shape[1]//2  # Number of top features to select initially for MTOPSIS, adjust as needed
    top_features_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Apply MTOPSIS to get the ranking criterion
    logging.info("Applying MTOPSIS for feature ranking...")
    RC = MTOPSIS(X_train, y_train, λ)
    
    # Select top features based on the ranking criterion
    logging.info("Selecting top features...")
    top_indices = select_top_features(X_train, RC, f)
    print(top_indices)
    
    # Select data corresponding to the top indices
    selected_data_train = X_train[:, top_indices]
    selected_data_test = X_test[:, top_indices]
    
    # Initialize the Modified ACO
    logging.info("Initializing Modified ACO...")
    aco = ModifiedACO(no_ants, no_cycles, n_features, q0, rho, beta_param)
    
    # Fit the ACO to select features for different percentages of top features
    logging.info("Fitting ACO to select features for different percentages of top features...")
    total_features = selected_data_train.shape[1]
    
    selected_indices = aco.fit(selected_data_train, y_train)
    
    for percentage in top_features_percentages:
        # Select features for KNN based on ACO results for the maximum percentage
        logging.info("Selecting features for KNN based on ACO results...")
        logging.info(f"Top {percentage}% selected features: {selected_indices}")
        num_top_features = max(1, total_features * percentage // 100)
        top_selected_indices = selected_indices[:num_top_features]
    
        # Logging the selected features
        logging.info("Selecting features for KNN based on ACO results...")
        logging.info(f"Top {percentage}% selected features: {top_selected_indices}")
        
        # Select features from the training and testing data
        final_selected_features_train = selected_data_train[:, top_selected_indices]
        final_selected_features_test = selected_data_test[:, top_selected_indices]
        # Apply multi-output KNN on the final selected features
        logging.info("Applying MultiOutputClassifier with KNN on the final selected features...")
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        multi_output_knn = MultiOutputClassifier(knn)
        multi_output_knn.fit(final_selected_features_train, y_train)
        logging.info("Multi-output KNN Training Complete. Model is ready to make predictions.")

        # Make predictions on the test set
        logging.info("Making predictions on the test set...")
        predictions = multi_output_knn.predict(final_selected_features_test)

        
        # Evaluate the model using various metrics
        logging.info("Evaluating the model using various metrics...")
        
        # Accuracy
        logging.info("Calculating accuracy...")
        accuracy_score_value = accuracy_score(y_test, predictions)
        logging.info(f"Accuracy score: {accuracy_score_value}")

        # Hamming Loss
        logging.info("Calculating hamming loss...")
        hamming_loss_score = hamming_loss(y_test, predictions)
        logging.info(f"Hamming loss score: {hamming_loss_score}")

        # One Error
        logging.info("Calculating one error...")
        one_error_score = one_error(y_test, predictions)
        logging.info(f"One error score: {one_error_score}")

        # Ranking Loss
        logging.info("Calculating ranking loss...")
        ranking_loss_score = ranking_loss(y_test, predictions)
        logging.info(f"Ranking loss score: {ranking_loss_score}")

        # Log metrics to wandb
        logging.info("Logging metrics to wandb...")
        try:
            # Initialize a new wandb run
            logging.info("Initializing a new wandb run...")
            wandb.init(project="your_project_name")  # Specify your project name if needed
            logging.info("wandb run initialized.")

            # Log metrics
            logging.info("Logging dataset name...")
            wandb.log({"dataset": dataset_name})
            logging.info(f"Dataset name '{dataset_name}' logged to wandb.")

            logging.info("Logging predictions...")
            wandb.log({"predictions": predictions.tolist()})
            logging.info("Predictions logged to wandb.")

            logging.info("Logging accuracy score...")
            wandb.log({"accuracy": accuracy_score_value})
            logging.info(f"Accuracy score '{accuracy_score_value}' logged to wandb.")

            logging.info("Logging hamming loss score...")
            wandb.log({"hamming_loss": hamming_loss_score})
            logging.info(f"Hamming loss score '{hamming_loss_score}' logged to wandb.")

            logging.info("Logging one error score...")
            wandb.log({"one_error": one_error_score})
            logging.info(f"One error score '{one_error_score}' logged to wandb.")

            logging.info("Logging ranking loss score...")
            wandb.log({"ranking_loss": ranking_loss_score})
            logging.info(f"Ranking loss score '{ranking_loss_score}' logged to wandb.")

            logging.info("Metrics logged to wandb.")
            
        except Exception as e:
            logging.error(f"An error occurred while logging metrics to wandb: {e}")

    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import logging

    # List of datasets
    datasets = [
        'Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron',
        'genbase', 'mediamill', 'medical', 'rcv1subset1', 'rcv1subset2',
        'rcv1subset3', 'rcv1subset4', 'rcv1subset5', 'scene', 'tmc2007_500', 'yeast'
    ]
    # Ensure logging is configured
    logging.basicConfig(level=logging.INFO)
    for dataset_name in datasets:
        logging.info(f"Processing dataset: {dataset_name}")
        main(dataset_name)
        logging.info(f"Completed processing for dataset: {dataset_name}")
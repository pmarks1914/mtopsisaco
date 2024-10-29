
import numpy as np
from scipy.stats import friedmanchisquare
from pprint import pprint

def evaluate_algorithm_performance(my_metrics, paper_metrics):
    metrics_results = {}
    total_wins, total_losses, total_ties = 0, 0, 0

    for metric in my_metrics:
        # Use duplicated arrays to meet Friedman's test requirement
        # This is only for demonstration; replace this with real additional samples if possible.
        my_data = np.tile(my_metrics[metric], (3, 1))
        paper_data = np.tile(paper_metrics[metric], (3, 1))

        # Calculate p-value using Friedman's test
        p_value = friedmanchisquare(my_data[0], my_data[1], paper_data[0])[1]

        # Calculate win/loss/tie for each metric
        wins = (my_metrics[metric] > paper_metrics[metric]).sum()
        losses = (my_metrics[metric] < paper_metrics[metric]).sum()
        ties = (my_metrics[metric] == paper_metrics[metric]).sum()

        metrics_results[metric] = {
            "p_value": p_value,
            "wins": wins,
            "losses": losses,
            "ties": ties
        }

        # Aggregate a single win/loss/tie based on majority for each metric
        if wins > losses:
            total_wins += 1
        elif losses > wins:
            total_losses += 1
        else:
            total_ties += 1

    # Final win/loss/tie counts across all metrics
    overall_results = {
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_ties": total_ties
    }

    return metrics_results, overall_results

# Example usage with additional metrics
my_metrics = {
    'accuracy': np.array([0.11]),
    'hamming_loss': np.array([0.014]),
    'coverage': np.array([139.02]),
    'precision': np.array([0.22]),
    'one_error': np.array([0.0]),
    'ranking_loss': np.array([0.78]),
}

paper_metrics = {
    'accuracy': np.array([0.27]),
    'hamming_loss': np.array([0.08]),
    'coverage': np.array([0.40]),
    'precision': np.array([0.58]),
    'one_error': np.array([0.48]),
    'ranking_loss': np.array([0.16]),
}

results, overall_results = evaluate_algorithm_performance(my_metrics, paper_metrics)
print("Detailed Results by Metric:", results)
print("Overall Win/Loss/Tie:", overall_results)

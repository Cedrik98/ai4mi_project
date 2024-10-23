import json
import math
from statistics import mean, stdev
import os

def calculate_metrics(filepath):
    """Calculates the mean and standard deviation of metrics for a given JSON file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # Extract overall patient metrics
    patient_metrics = [patient['overall'] for patient in data['patients'].values()]
    
    metrics_results = {}
    for metric in patient_metrics[0].keys():
        values = [patient[metric] for patient in patient_metrics]
        metrics_results[metric] = {
            'mean': mean(values),
            'std': stdev(values)
        }
    
    return metrics_results

def calculate_metrics_for_all_json(directory):
    """Iterates through all JSON files in the directory and calculates metrics."""
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            metrics = calculate_metrics(filepath)
            
            # Print the mean and standard deviation results for each metric
            for metric, result in metrics.items():
                print(f"Metric: {metric}")
                print(f"  Mean: {result['mean']}")
                print(f"  Standard deviation: {result['std']}")
            print("\n")  

# Specify the directory containing all the JSON files
directory = 'all_metrics'

# Process each JSON file in the folder
calculate_metrics_for_all_json(directory)

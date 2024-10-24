import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metrics(data, metrics):
    results = []
    for patient, patient_data in data['patients'].items():
        for metric in metrics:
            results.append({
                'Patient': patient,
                'Metric': metric,
                'Value': patient_data['overall'][metric]
            })
    return results

def create_dataframe(model_data):
    all_results = []
    for model, data in model_data.items():
        results = extract_metrics(data, metrics_to_plot)
        for result in results:
            result['Model'] = model
        all_results.extend(results)
    return pd.DataFrame(all_results)

def plot_boxplots(df, output_folder):
    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(models_to_compare))

    for metric in metrics_to_plot:
        _, ax = plt.subplots(figsize=(12, 8))
        
        # Create boxplot
        sns.boxplot(x='Model', y='Value', data=df[df['Metric'] == metric], 
                    palette=cmap(np.linspace(0, 1, len(models_to_compare))),
                    width=0.6, linewidth=1.5, ax=ax)

        # Customize the plot
        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Models', fontsize=20, fontweight='bold')
        plt.ylabel(metric.replace("_", " ").title(), fontsize=16)
        plt.xlabel('Model', fontsize=16)
        plt.xticks(rotation=0, fontsize=14)
        plt.yticks(fontsize=14)

        # Add a grid for better readability
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        ax.set_facecolor('#F8F8F8')
        
        medians = df[df['Metric'] == metric].groupby('Model')['Value'].median()
        vertical_offset = df[df['Metric'] == metric]['Value'].max() * 0.01  # offset for median labels

        for xtick in ax.get_xticks():
            ax.text(xtick, medians[xtick] + vertical_offset, f'{medians[xtick]:.3f}', 
                    horizontalalignment='center', size='x-small', color='dimgrey', weight='semibold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{metric}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Folder where json files are saved
json_folder = 'results/segthor/all_metrics'  
output_folder = 'results/segthor/plots' 
metrics_to_plot = ['dice_score', 'hausdorff_distance', 'average_surface_distance', 'volumetric_similarity', 'false_negative_rate']  
models_to_compare = ['ce_base', 'ce_rotated' ,'ce_elastic', 'ce_gaussian', 'ce_threshold']  

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load data for each model
model_data = {}
for model in models_to_compare:
    json_path = os.path.join(json_folder, f'{model}.json')
    if os.path.exists(json_path):
        model_data[model] = load_json(json_path)
    else:
        print(f"Warning: JSON file for {model} not found.")

df = create_dataframe(model_data)
plot_boxplots(df, output_folder)
print(f"Plots saved in {output_folder}")

"""
Script visualizing the non-IID data distribution across clients.
Generates a heatmap/bar chart showing sample counts per class for each client.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocessing import load_and_preprocess
from data.partition import partition_data_non_iid, assign_clients_to_edges

def visualize_distribution(dataset_name='nf_ton_iot', num_clients=60, num_edges=3, 
                           alpha_benign=0.3, alpha_attack=0.8, max_samples=100000):
    print(f"Loading {dataset_name} (max {max_samples} samples)...")
    X_train, X_test, y_train, y_test, scaler, label_map = load_and_preprocess(
        './data/raw/', dataset_name=dataset_name, 
        test_size=0.4, max_samples=max_samples
    )
    
    # Dirichlet alpha per class (Paper: 0.3 for Benign, 0.8 for Attack)
    alpha_dict = {0: alpha_benign}
    for c in np.unique(y_train):
        if c != 0:
            alpha_dict[int(c)] = alpha_attack
            
    print(f"Partitioning data (alpha_benign={alpha_benign}, alpha_attack={alpha_attack})...")
    client_data_indices, client_classes = partition_data_non_iid(
        y_train, num_clients=num_clients, alpha=alpha_dict, seed=2024
    )
    edge_client_map = assign_clients_to_edges(num_clients, num_edges)
    
    # Create a distribution matrix: (clients x classes)
    print(f"Label Map: {label_map}")
    num_classes = len(label_map)
    dist_matrix = np.zeros((num_clients, num_classes))
    
    for client_id, indices in client_data_indices.items():
        y_client = y_train[indices]
        classes, counts = np.unique(y_client, return_counts=True)
        for cls, count in zip(classes, counts):
            if int(cls) < num_classes:
                dist_matrix[client_id, int(cls)] = count
            
    # Convert to DataFrame for easier plotting
    inverse_label_map = {v: k for k, v in label_map.items()}
    class_names = [inverse_label_map.get(i, f"Class {i}") for i in range(num_classes)]
    
    df = pd.DataFrame(dist_matrix, columns=class_names)
    df.index.name = 'Client ID'
    
    # Plotting Heatmap (only show first 20 clients if N=60 to keep it readable, but save all in bar chart)
    plt.figure(figsize=(16, 10))
    # We plot a subset of clients for the heatmap to avoid clutter
    subset_n = min(20, num_clients)
    sns.heatmap(df.iloc[:subset_n], annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Number of Samples'})
    
    # Add Edge Server grouping info to Y-axis labels
    new_labels = []
    for cid in range(subset_n):
        # Find which edge this client belongs to
        edge_id = -1
        for eid, cids in edge_client_map.items():
            if cid in cids:
                edge_id = eid
                break
        new_labels.append(f"C{cid} (E{edge_id})")
    
    plt.yticks(np.arange(subset_n) + 0.5, new_labels, rotation=0)
    plt.title(f"Data Distribution (Heatmap: subset of {subset_n}/{num_clients} clients)\nNon-IID, alpha_benign={alpha_benign}, alpha_attack={alpha_attack}")
    plt.tight_layout()
    
    # Create output directory for plots
    os.makedirs('./plots', exist_ok=True)
    out_path = f'./plots/{dataset_name}_distribution.png'
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")
    
    # Generate a horizontal stacked bar chart (showing ALL clients)
    plt.figure(figsize=(16, 12))
    df_pct = df.div(df.sum(axis=1), axis=0) * 100
    df_pct.plot(kind='barh', stacked=True, colormap='tab20', figsize=(16, 12))
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Percentage of Samples (%)')
    plt.title(f"Class Distribution % (ALL {num_clients} Clients) - {dataset_name}\nalpha_benign={alpha_benign}, alpha_attack={alpha_attack}")
    
    # Y-labels for all 60 clients might be crowded, so let's label every 5th or similar
    all_new_labels = []
    for cid in range(num_clients):
        edge_id = -1
        for eid, cids in edge_client_map.items():
            if cid in cids:
                edge_id = eid; break
        all_new_labels.append(f"C{cid} (E{edge_id})")
    
    tick_pos = np.arange(num_clients)
    plt.yticks(tick_pos, all_new_labels, fontsize=8)
    plt.tight_layout()
    
    out_path_bar = f'./plots/{dataset_name}_distribution_bar.png'
    plt.savefig(out_path_bar)
    print(f"Bar chart saved to {out_path_bar}")
    
    return out_path, out_path_bar

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nf_ton_iot')
    parser.add_argument('--num_clients', type=int, default=60)
    parser.add_argument('--num_edges', type=int, default=3)
    parser.add_argument('--alpha_benign', type=float, default=0.3)
    parser.add_argument('--alpha_attack', type=float, default=0.8)
    parser.add_argument('--max_samples', type=int, default=100000)
    args = parser.parse_args()
    
    visualize_distribution(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        num_edges=args.num_edges,
        alpha_benign=args.alpha_benign,
        alpha_attack=args.alpha_attack,
        max_samples=args.max_samples
    )

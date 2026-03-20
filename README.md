# HFIN: Hierarchical Federated Class-Incremental Learning for NIDS

HFIN is a robust deep learning framework designed to address the challenges of Network Intrusion Detection Systems (NIDS) in environments with constantly evolving threats. It combines **Federated Learning** for collaborative model training across multiple clients without compromising data privacy, and **Class-Incremental Learning** to adapt the models to new, unseen attack types over time without catastrophic forgetting.

## Key Features
*   **Hierarchical Architecture**: Employs an Edge-Cloud hierarchical federated setup to optimize communication and model aggregation.
*   **Class-Incremental Learning**: Integrates strategies to incrementally learn new anomaly classes while retaining knowledge of previous classes.
*   **Privacy-Preserving**: Client data remains local; only model parameters are shared during training.

## Project Structure
*   `config/`: Configuration settings for the framework.
*   `data/`: Scripts and datasets for data preprocessing and partitioning among clients.
*   `models/`: Core neural network architectures and feature extractors.
*   `federated/`: Implementation of the federated learning logic (clients, edge servers, cloud server).
*   `incremental/`: Modules handling class-incremental learning strategies (e.g., distillation, exemplars).
*   `main.py`: The entry point for running experiments.
*   `evaluate.py`: Script for evaluating the global model's performance.

## Installation & Requirements
Check `requirements.txt` for all necessary dependencies. Typically requires:
*   Python 3.8+
*   PyTorch
*   Pandas, NumPy, Scikit-learn

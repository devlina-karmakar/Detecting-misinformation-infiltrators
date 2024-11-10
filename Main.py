from data_preparation import load_data, initialize_graph, add_edges_with_weights
from community_detection import detect_communities
from feature_extraction import extract_features_labels, add_synthetic_labels
from resampling import balance_dataset
from train_test_split import prepare_data
from model import GRUMisinformationDetector
from train_evaluate import train_model, evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim

# Load data
file_path = 'final_community_dataset_with_attributes.csv'
data = load_data(file_path)

# Select the community of interest
community_name = 'WBJDF'
community_nodes = data[data['community'] == community_name]

# Initialize graph and add nodes and edges
G, limited_nodes = initialize_graph(community_nodes)
G = add_edges_with_weights(G, limited_nodes)

# Community detection
G, community_map = detect_communities(G)

# Feature extraction
features, labels = extract_features_labels(G, community_map)
labels = add_synthetic_labels(labels)

# Resampling
X_resampled, y_resampled = balance_dataset(features, labels)

# Data preparation
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = prepare_data(X_resampled, y_resampled)

# Initialize model
input_size = X_train_tensor.shape[1]
hidden_size = 64
output_size = 2
model = GRUMisinformationDetector(input_size, hidden_size, output_size)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor)

# Evaluation
evaluate_model(model, X_test_tensor, y_test_tensor)

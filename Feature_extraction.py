import numpy as np
from collections import Counter

def extract_features_labels(G, community_map, label_col='misinformation_spreader'):
    features = []
    labels = []

    for node, attrs in G.nodes(data=True):
        feature_vector = [
            attrs['age'],
            attrs['gender'],
            attrs['activity'],
            attrs['reaction'],
            attrs['post per day'],
            attrs['group vs personal post'],
            community_map[node]
        ]
        features.append(feature_vector)
        labels.append(attrs.get(label_col, 0))

    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def add_synthetic_labels(labels, ratio=0.1):
    label_counts = Counter(labels)
    if label_counts[0] > label_counts[1]:
        non_spreader_indices = np.where(labels == 0)[0]
        np.random.seed(11)
        synthetic_spreaders = np.random.choice(non_spreader_indices, int(ratio * len(non_spreader_indices)), replace=False)
        labels[synthetic_spreaders] = 1
    return labels

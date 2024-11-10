# Detecting-misinformation-infiltrators
# Misinformation Spreader Detection with Graph Neural Networks

## Project Overview

This project implements a pipeline to detect misinformation spreaders within a community graph using a GRU-based classifier. It involves:
- Loading and preparing data
- Building a graph based on community interactions
- Detecting communities using Label Propagation
- Extracting features and labels for classification
- Handling data imbalance using SMOTE
- Training a GRU-based neural network for binary classification

---

## Prerequisites

The project requires the following Python packages:

```bash
numpy
pandas
networkx
scikit-learn
imblearn
torch

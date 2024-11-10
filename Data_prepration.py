import pandas as pd
import numpy as np
import networkx as nx

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def initialize_graph(community_data, node_limit=1100):
    G = nx.Graph()
    limited_nodes = community_data.iloc[:node_limit]
    for _, row in limited_nodes.iterrows():
        G.add_node(row['node_id'], **row.to_dict())
    return G, limited_nodes

def add_edges_with_weights(G, limited_nodes):
    for i, node1 in limited_nodes.iterrows():
        for j, node2 in limited_nodes.iterrows():
            if i < j:
                interaction_score = (
                    (node1['reaction'] + node2['reaction']) / 2 +
                    (node1['post per day'] * node1['group vs personal post']) +
                    (node2['post per day'] * node2['group vs personal post'])
                )
                G.add_edge(node1['node_id'], node2['node_id'], weight=interaction_score)
    return G

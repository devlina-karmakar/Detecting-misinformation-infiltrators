import networkx as nx
from networkx.algorithms.community import label_propagation_communities

def detect_communities(G):
    communities = label_propagation_communities(G)
    community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    nx.set_node_attributes(G, community_map, 'community')
    return G, community_map

"""
Author: Jens van Drunen (572793fd)
Affiliation: Erasmus University Rotterdam
Email: 572793fd@eur.nl
Description: graph_drawing.py - Contains functionalities for constructing and visualising graphs.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def constructGraph_AND(neighborhoods):
    """
    Constructs an undirected graph using the AND-rule for neighborhood inclusion.
    An edge is added if both nodes are present in each other's neighborhood lists.
    
    Parameters:
    neighborhoods (dict): Dictionary with nodes as keys and lists of neighboring nodes as values.
    
    Returns:
    nx.Graph: Constructed graph.
    """
    
    edges = []
    nodes = set(neighborhoods.keys())

    # AND-rule
    for node, neighbors in neighborhoods.items():
        for neighbor in neighbors:
            if node in neighborhoods[neighbor]:
                edges.append((node, neighbor))
      
    # Initialise graph
    G = nx.Graph()
    G.add_nodes_from(nodes)    
    G.add_edges_from(edges)
    
    # Increase figure size and resolution
    plt.figure(figsize=(8, 4), dpi=150)
    
    # Use a layout algorithm to space nodes more evenly
    pos = nx.spring_layout(G, k=4, iterations=500)

    # Draw graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
    
    print("Nodes (AND):", G.nodes)
    print("Edges (AND):", G.edges)
    
    plt.show()
    
    return G


def constructGraph_AND_bincat(neighborhoods, categorical_vars, binary_vars):
    """
    Constructs an undirected graph using the AND-rule for neighborhood inclusion and colors nodes 
    based on whether they are binary or categorical variables.

    Parameters:
    neighborhoods (dict): Dictionary with nodes as keys and lists of neighboring nodes as values.
    categorical_vars (list): List of categorical variable names.
    binary_vars (list): List of binary variable names.

    Returns:
    nx.Graph: Constructed graph with colored nodes.
    """
    
    edges = []
    nodes = set(neighborhoods.keys())
    
    # AND-rule
    for node, neighbors in neighborhoods.items():
        for neighbor in neighbors:
            if node in neighborhoods[neighbor]:
                edges.append((node, neighbor))
                

   # Initialise graph
    G = nx.Graph()
    G.add_nodes_from(nodes)    
    G.add_edges_from(edges)
    
    color_map = []
    for node in G:
        if node in binary_vars:
            color_map.append('#ff9999')  # Light red for binary variables
        elif any(node in cat_var for cat_var in categorical_vars):
            color_map.append('#99ccff')  # Light blue for categorical variables
        else:
            color_map.append('lightgray')  # Default color for nodes not in the lists
    
    # Draw the graph
    plt.figure(figsize=(10, 6), dpi=150)
    pos = nx.spring_layout(G, k=8, iterations=700)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=color_map, font_size=8, font_color='black', font_weight='bold')
    
    # Create a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9999', markersize=10, label='Binary Variables'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#99ccff', markersize=10, label='Categorical Variables')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    print("Nodes (AND):", G.nodes)
    print("Edges (AND):", G.edges)
    
    plt.show()
    
    return G


def constructGraph_OR(neighborhoods):
    """
    Constructs an undirected graph using the OR-rule for neighborhood inclusion.
    An edge is added if either node is present in the other's neighborhood list.

    Parameters:
    neighborhoods (dict): Dictionary with nodes as keys and lists of neighboring nodes as values.

    Returns:
    nx.Graph: Constructed graph.
    """
    
    edges = []
    nodes = set(neighborhoods.keys())
    
    #OR-rule
    for node, neighbors in neighborhoods.items():
        for neighbor in neighbors:
            if node in neighborhoods[neighbor] or neighbor in neighborhoods[node]:
                edges.append((node, neighbor))
    
    #Initialise graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    
    # Increase figure size and resolution
    plt.figure(figsize=(8, 4), dpi=150)
    
    # Use a layout algorithm to space nodes more evenly
    pos = nx.spring_layout(G, k=4, iterations=500)
    
    # Draw the graph with the shorter labels
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
    
    print("Nodes (AND):", G.nodes)
    print("Edges (AND):", G.edges)
    
    plt.show()
    
    return G


def create_graph_from_structure_matrix(matrix, node_order, categorical_vars, binary_vars):
    """
    Creates and plots a graph from a predefined conditional dependence structure matrix.

    Parameters:
    matrix (np.array): Predefined conditional dependence structure matrix.
    node_order (list): List of node names in the order corresponding to the matrix.
    categorical_vars (list): List of categorical variable names.
    binary_vars (list): List of binary variable names.

    Returns:
    nx.Graph: Graph constructed from the matrix.
    """
    
    # Ensure matrix is a numpy array
    matrix = np.array(matrix)
    
    # Initialise graph
    G = nx.Graph()
    G.add_nodes_from(node_order)
    
    # Add edges based on the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                G.add_edge(node_order[i], node_order[j])
    
    # Initialise color
    color_map = []
    for node in G:
        if node in binary_vars:
            color_map.append('#ff9999')  # Light red for binary variables
        elif any(node in cat_var for cat_var in categorical_vars):
            color_map.append('#99ccff')  # Light blue for categorical variables
        else:
            color_map.append('lightgray')  # Default color for nodes not in the lists
    
    # Draw the graph
    plt.figure(figsize=(12, 8), dpi=150)
    pos = nx.spring_layout(G, k=5, iterations=600)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=color_map, font_size=8, font_color='black', font_weight='bold')
    
    # Create a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9999', markersize=10, label='Binary Variables'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#99ccff', markersize=10, label='Categorical Variables')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.show()
    
    return G

def reconstruct_neighborhoods(nodes, edges):
    """
    Reconstructs the neighborhood mapping from nodes and edges.

    Parameters:
    nodes (list): List of nodes.
    edges (list): List of edges (tuples).

    Returns:
    dict: Reconstructed neighborhood mapping.
    """
    
    neighborhoods = {node: [] for node in nodes}
    
    for edge in edges:
        node1, node2 = edge
        neighborhoods[node1].append(node2)
        neighborhoods[node2].append(node1)
    
    return neighborhoods
"""
Author: Jens van Drunen (572793fd)
Affiliation: Erasmus University Rotterdam
Email: 572793fd@eur.nl
Description: evaluation.py - Contains functionalities for evaluating the performance of identified graphical structures.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import logging

# Setup logging
logger = logging.getLogger(__name__)

def create_known_structure_matrix_simBasic():
    """
    Creates the known structure matrix for the basic simulation.

    Returns:
    np.array: Known structure matrix.
    """
    
    known_structure = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1]
])
    
    return known_structure

def create_known_structure_matrix_SMCsim_categories():
    """
    Creates the known structure matrix for the SMC simulation with categories.

    Returns:
    np.array: Known structure matrix.
    """
    
    known_structure = np.zeros((8, 8), dtype=int)
    
    # Cancertype and Hist
    known_structure[0,1] = 1
    known_structure[1,0] = 1
    
    # Pam50 and Concens 
    known_structure[2,5] = 1
    known_structure[5,2] = 1
    
    # Tstage and Purity
    known_structure[3,4] = 1
    known_structure[4,3] = 1
    
    # Inconst and Chemo
    known_structure[2,7] = 1
    known_structure[7,2] = 1
    
    # Meno and chemo
    known_structure[6,7] = 1
    known_structure[7,6] = 1
    
    return known_structure
       

def create_known_structure_matrix_METABRICsim_categories():
    """
    Creates the known structure matrix for the METABRIC simulation with categories.

    Returns:
    np.array: Known structure matrix.
    """
    
    known_structure = np.zeros((18, 18), dtype=int)
    
    # Cancer type dependent on histo
    known_structure[0,4] = 1
    known_structure[4,0] = 1
    
    # Neoplasm dependent on pam50, cellularity and tstage
    known_structure[3,1] = 1
    known_structure[3,2] = 1
    known_structure[3,6] = 1
    known_structure[1,3] = 1
    known_structure[2,3] = 1
    known_structure[6,3] = 1
    
    # Tumor stage dependent on breastsurgery, chemo and relapsefree
    known_structure[6,15] = 1
    known_structure[6,9] = 1
    known_structure[6,16] = 1
    known_structure[15,6] = 1
    known_structure[16,6] = 1
    known_structure[9,6] = 1
    
    # Relapse free dependent on vital status
    known_structure[16,7] = 1
    known_structure[7,16] = 1
    
    # HER2 dependent on hormone, chemo, radio, 3Gene
    known_structure[13,10] = 1
    known_structure[13,9] = 1
    known_structure[13,11] = 1
    known_structure[13,5] = 1
    known_structure[10,13] = 1
    known_structure[9,13] = 1
    known_structure[11,13] = 1
    known_structure[5,13] = 1
    
    # ER dependent on chemo, hormone, 3Gene
    known_structure[14,9] = 1
    known_structure[14,10] = 1
    known_structure[14,5] = 1
    known_structure[9,14] = 1
    known_structure[10,14] = 1
    known_structure[5,14] = 1
    
    # PR dependent on radio
    known_structure[12,11] = 1
    known_structure[11,12] = 1
    
    # Meno dependent on hormone
    known_structure[8,10] = 1
    known_structure[10,8] = 1

    return known_structure

def create_identified_structure_matrix(G, node_order):
    """
    Creates a structure matrix from a graph.

    Parameters:
    G (nx.Graph): Graph from which to create the structure matrix.
    node_order (list): List of nodes in the order they should appear in the matrix.

    Returns:
    np.array: Structure matrix.
    """
    
    matrix = nx.to_numpy_array(G, nodelist=node_order)
    return matrix.astype(int)
    

def evaluate_SDH_YOUDEN(known_structure, identified_structure):
    """
    Evaluates the identified structure against the known structure using SDH and Youden's Index.

    Parameters:
    known_structure (np.array): Known structure matrix.
    identified_structure (np.array): Identified structure matrix.

    Returns:
    tuple: SDH (Sum of Distance to Hypothesis), Youden's Index.
    """
    
    if known_structure.shape != identified_structure.shape:
        raise ValueError("Known structure and identified structure must have the same shape")
    
    n = known_structure.shape[0]
    
    FP = FN = TP = TN = 0
    for i in range(n):
        for j in range(i + 1, n):
            if known_structure[i, j] == 1 and identified_structure[i, j] == 1:
                TP += 1
            elif known_structure[i, j] == 0 and identified_structure[i, j] == 1:
                FP += 1
            elif known_structure[i, j] == 1 and identified_structure[i, j] == 0:
                FN += 1
            elif known_structure[i, j] == 0 and identified_structure[i, j] == 0:
                TN += 1
                
    SDH = FP + FN
    
    sensivity = TP / (TP + FN) if (TP+FN)>0 else 0 
    specificity = TN / (TN + FP) if (TN+FP)>0 else 0 
    
    YOUDEN = sensivity + specificity - 1
    
    return SDH, YOUDEN
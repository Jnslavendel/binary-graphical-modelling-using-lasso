"""
Author: Jens van Drunen
Affiliation: Erasmus University Rotterdam
Email: 572793fd@eur.nl
Description: data_processing.py - Contains functionalities for the processing of the emperical SMC and METABRIC datasets
"""

import pandas as pd


def load_data(file_path):
    """
    Load data from a .tsv file.

    Parameters:
    file_path (str): The path to the .tsv file.

    Returns:
    pd.DataFrame: The loaded data as a Pandas DataFrame.
    """
    data = pd.read_csv(file_path, sep='\t')
    return data


def preprocess_data_METABRIC(data):
    """
    Preprocess the METABRIC dataset by making specified variables binary using one-hot encoding.

    Parameters:
    data (pd.DataFrame): The data to preprocess.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    
    # Drop specified columns 
    columns_to_drop = [
        "Study ID", "Patient ID", "Sample ID", "Cancer Type",
        "ER status measured by IHC", "HER2 status measured by SNP6",
        "Number of Samples Per Patient", "Sample Type", "Sex", "Oncotree Code", "Integrative Cluster", "Cohort"
    ]
    data = data.drop(columns=columns_to_drop)
    
    # Columns to make binary 
    binary_columns = [
        "Type of Breast Surgery", "Cancer Type Detailed",
        "Cellularity", "Chemotherapy", "Pam50 + Claudin-low subtype",
        "ER Status", "Neoplasm Histologic Grade", "HER2 Status",
        "Tumor Other Histologic Subtype", "Hormone Therapy", 
        "Inferred Menopausal State", "Primary Tumor Laterality", 
        "Overall Survival Status", "PR Status", "Radio Therapy",
        "Relapse Free Status", "3-Gene classifier subtype", 
        "Tumor Stage", "Patient's Vital Status"
    ]
    
    # Create dummy variables using one-hot encoding
    data = pd.get_dummies(data, columns=binary_columns, drop_first=False)
    
    # Convert boolean values to integers
    for column in data.columns:
        if data[column].dtype == 'bool':
            data[column] = data[column].astype(int)
    
    # Drop rows with any missing values
    data = data.dropna()
    
    # Ensure each categorical variable has exactly one dummy with a value of 1 per row
    prefix_set = set([col.split('_')[0] for col in data.columns if '_' in col])
    valid_rows = []
    invalid_rows = []
    for index, row in data.iterrows():
        valid = True
        for prefix in prefix_set:
            dummies = [col for col in data.columns if col.startswith(prefix + '_')]
            dummy_sum = row[dummies].sum()
            if dummy_sum != 1:
                valid = False
                break
        if valid:
            valid_rows.append(row)
        else:
            invalid_rows.append(row) 
    
    data = pd.DataFrame(valid_rows, columns=data.columns)
 
    invalid_data = pd.DataFrame(invalid_rows, columns=data.columns)
    invalid_data.to_excel("invalid_rows_METABRIC.xlsx", index=False)
    
    # Identify binary columns with fewer than 5 ocurrences 
    columns_to_remove = []
    for column in data.columns:
        if data[column].dtype in [int, float]:
            if data[column].sum() < 6:
                columns_to_remove.append(column)
                
    # Remove rows where the columns to be dropped have '1's
    for column in columns_to_remove:
        data = data[data[column] != 1]
    
    # Logging: print columns to be dropped
    for column in columns_to_remove:
        print(f"Dropped column: {column} - Number of '1's: {data[column].sum()}")
    
    # Remove the identified columns
    data = data.drop(columns=columns_to_remove)
    
    # Logging: print final columns and row count
    print(f"Number of rows after processing: {len(data)}")
    
    return data


def preprocess_data_SMC(data):
    """
    Preprocess the SMC dataset by making specified variables binary using one-hot encoding.

    Parameters:
    data (pd.DataFrame): The data to preprocess.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    
    # Drop specified columns 
    columns_to_drop = [
        "Study ID", "Patient ID", "Sample ID", "Cancer Type","Immunohistochemistry Subtype",
        "Center of sequencing", "Cohort", "Oncotree Code", "Race Category", "Sample Class", "Number of Samples Per Patient", "Sample Type", "Sex", "Somatic Status"
    ]
    data = data.drop(columns=columns_to_drop)
    
    # Columns to make binary 
    binary_columns = [
        "Cancer Type Detailed", "Histology ", "TNM Stage", "Menopausal Status At Diagnosis",
        "PAM50 subtype", "Subtype Consensus", "Chemotherapy"
    ]

     # Create dummy variables using one-hot encoding
    data = pd.get_dummies(data, columns=binary_columns, drop_first=False)
    
    # Convert boolean values to integers
    for column in data.columns:
        if data[column].dtype == 'bool':
            data[column] = data[column].astype(int)
    
    # Drop rows with any missing values
    data = data.dropna()
    
    # Ensure each categorical variable has exactly one dummy with a value of 1 per row
    prefix_set = set([col.split('_')[0] for col in data.columns if '_' in col])
    valid_rows = []
    invalid_rows = []  
    for index, row in data.iterrows():
        valid = True
        for prefix in prefix_set:
            dummies = [col for col in data.columns if col.startswith(prefix + '_')]
            dummy_sum = row[dummies].sum()
            if dummy_sum != 1:
                valid = False
                break
        if valid:
            valid_rows.append(row)
        else:
            invalid_rows.append(row) 
    
    # Construct valid data
    data = pd.DataFrame(valid_rows, columns=data.columns)
    
    # Logging: export invalid data
    invalid_data = pd.DataFrame(invalid_rows, columns=data.columns)
    invalid_data.to_excel("invalid_rows_SMC.xlsx", index=False)
    
    # Identify binary columns with fewer than 5 ocurrences 
    columns_to_remove = []
    for column in data.columns:
        if data[column].dtype in [int, float]:
            if data[column].sum() < 5:
                columns_to_remove.append(column)
                
    # Remove rows where the columns to be dropped have '1's
    for column in columns_to_remove:
        data = data[data[column] != 1]
    
    # Logging: print columns to be dropped 
    for column in columns_to_remove:
        print(f"Dropped column: {column} - Number of '1's: {data[column].sum()}")
    
    # Remove the identified columns
    data = data.drop(columns=columns_to_remove)
    
    # Logging: print final columns and row count
    print(f"Number of rows after processing: {len(data)}")
    
    return data
    
def get_binary_variables(data):
    """
    Retrieve all binary variables from a DataFrame.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.sss
    
    Returns:
    list: List of binary variable names.
    """
    binary_variables = [col for col in data.columns if data[col].nunique() == 2]
    return binary_variables


def rename_nodes_and_edges_METABRIC(nodes, edges, node_order):
    """
    Renames nodes and edges according to a specified order in emperical METABRIC

    Parameters:
    nodes (list): List of original node names.
    edges (list): List of tuples representing edges between nodes.
    node_order (list): List of new node names in the desired order.

    Returns:
    new_nodes (list): List of renamed nodes.
    new_edges (list): List of tuples representing renamed edges.
    """
    # Create a mapping from old node names to new node names
    node_mapping = {
        'Inferred': 'Menop', 'ER': 'ER', 'Chemotherapy': 'Chemo', 'Hormone': 'Hormo', 'Cellularity': 'Cellu',
        'Pam50': 'Pam50', 'Neoplasm': 'Neopl', 'Overall': 'SurvivalStatus', 'Radio': 'Radio', 'Relapse': 'Relap',
        '3-Gene': '3Gene', 'Patients': 'Vital', 'Type': 'Breas', 'Cancer': 'Cance', 'Primary': 'sideR',
        'HER2': 'HER2', 'PR': 'PR', 'Tumor': 'Tumor'
    }

    # Rename nodes using the mapping dictionary
    new_nodes = [node_mapping[node] for node in nodes]

    # Initialise an empty list to store renamed edges
    new_edges = []

    # Loop through each edge and rename the nodes in the edge
    for edge in edges:
        new_edge = (node_mapping[edge[0]], node_mapping[edge[1]])
        new_edges.append(new_edge)

    return new_nodes, new_edges



"""
Author: Jens van Drunen
Affiliation: Erasmus University Rotterdam
Email: 572793fd@eur.nl
Description: usage.py - Combines various simulation, modelling and evaluation functionalities for practical use
"""

import simulation 
import data_processing
import graphical_modelling
import graph_drawing
import evaluation 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm 
import os

def create_boxplots_basic_simulations(filepath='data-external/basicsim.xlsx'):
    """
    Creates boxplots for basic simulations based on SHD and YOUDEN metrics.

    Parameters:
    filepath (str): Path to the Excel file containing the simulation results.
    """
    
    #Load the data from the Excel file
    df_basic = pd.read_excel(filepath)

    # Rename columns for consistency
    df_basic.rename(columns={
         'SDH (AND)': 'SHD (AND)',
         'SDH (OR)': 'SHD (OR)'
     }, inplace=True)
     
    # Define the sample sizes
    sample_sizes = [100, 200, 500, 1000, 2000]
     
    # Split the dataframe into separate dataframes for each sample size
    dataframes = {size: df_basic[df_basic['Sample Size'] == size] for size in sample_sizes}
     
    # Function to melt the dataframe for easier plotting
    def melt_dataframe(df, metric, and_col, or_col):
        return df.melt(id_vars=["Method", "Replication", "Sample Size"], 
                       value_vars=[and_col, or_col], 
                       var_name="Rule", 
                       value_name=metric)
     
    # Plotting function with customisation for the median line
    def plot_boxplots(df, metric, and_col, or_col, sample_size):
         melted_df = melt_dataframe(df, metric, and_col, or_col)
         
         plt.figure(figsize=(12, 6))
         ax = sns.boxplot(x="Method", y=metric, hue="Rule", data=melted_df, showmeans=False,
                          medianprops={"color": "black", "linewidth": 2})  # Customize median line
         plt.title(f'{metric} for Sample Size {sample_size}')
         plt.ylabel(metric)
         plt.xlabel('Method')
         plt.legend(title='Rule')
         plt.xticks(rotation=45)
         plt.tight_layout()
         plt.show()
     
    # Update column names based on actual columns in dataset
    shd_and_col = "SHD (AND)"
    shd_or_col = "SHD (OR)"
    youden_and_col = "YOUDEN (AND)"
    youden_or_col = "YOUDEN (OR)"
     
    # Loop through the dataframes and create the boxplots for both metrics
    for size, df in dataframes.items():
         plot_boxplots(df, 'SHD', shd_and_col, shd_or_col, size)
         plot_boxplots(df, 'YOUDEN', youden_and_col, youden_or_col, size)


def create_boxplots_METABRIC_SMC_simulations(filepath_metabric='data-external/smc-sim-results.xlsx', filepath_smc='data-external/metabric-sim-results.xlsx'):
    """
    Creates boxplots for METABRIC and SMC simulations based on SHD and YOUDEN metrics.

    Parameters:
    filepath_metabric (str): Path to the Excel file containing the METABRIC simulation results.
    filepath_smc (str): Path to the Excel file containing the SMC simulation results.
    """
    
    # Load the correct data from the Excel files
    df_smc = pd.read_excel(filepath_metabric)
    df_metabric = pd.read_excel(filepath_smc)  
     
    # Check if the necessary columns are in the METABRIC dataframe
    expected_columns = ["Model", "SimMethod", "Replication", "Sample Size", "SHD (AND)", "YOUDEN (AND)", "SHD (OR)", "YOUDEN (OR)"]
    if not all(column in df_metabric.columns for column in expected_columns):
        raise ValueError("The METABRIC data file does not contain the expected columns for simulation results.")
     
    # Function to melt the dataframe for easier plotting
    def melt_dataframe(df, metric, and_col, or_col):
        return df.melt(id_vars=["Model", "SimMethod", "Replication", "Sample Size"], 
                        value_vars=[and_col, or_col], 
                        var_name="Rule", 
                        value_name=metric)
     
    # Plotting function with customisation for the median line
    def plot_boxplots(df, metric, and_col, or_col, dataset_name):
         melted_df = melt_dataframe(df, metric, and_col, or_col)
         sample_size = df['Sample Size'].unique()[0]
         
         plt.figure(figsize=(12, 6))
         ax = sns.boxplot(x="Model", y=metric, hue="Rule", data=melted_df, showmeans=False,
                          medianprops={"color": "black", "linewidth": 2})  # Customize median line
         plt.title(f'{metric} for {dataset_name} Simulation (n={sample_size})')
         plt.ylabel(metric)
         plt.xlabel('Model')
         plt.legend(title='Rule')
         plt.xticks(rotation=45)
         plt.tight_layout()
         plt.show()
     
    # Update column names based on actual columns in dataset
    shd_and_col = "SHD (AND)"
    shd_or_col = "SHD (OR)"
    youden_and_col = "YOUDEN (AND)"
    youden_or_col = "YOUDEN (OR)"
     
    # Plot SHD and YOUDEN Index boxplots for both datasets
    plot_boxplots(df_smc, 'SHD', shd_and_col, shd_or_col, 'SMC')
    plot_boxplots(df_metabric, 'SHD', shd_and_col, shd_or_col, 'METABRIC')
    plot_boxplots(df_smc, 'YOUDEN', youden_and_col, youden_or_col, 'SMC')
    plot_boxplots(df_metabric, 'YOUDEN', youden_and_col, youden_or_col, 'METABRIC')
     
  
def run_simulations(nsim, nsample, max_lambda, model, simulation_method, output_file):
   """
    Runs simulations and evaluates the results based on the selected model and simulation method.

    Parameters:
    nsim (int): Number of simulations to run.
    nsample (int): Sample size for each simulation.
    max_lambda (float): Maximum lambda value for LASSO.
    model (str): Model type ('lasso-cv', 'bolasso-cv', 'lasso-aic', 'lasso-bic').
    simulation_method (str): Simulation method ('basic', 'smc-categories').
    output_file (str): Path to the output file to save the results.
    """
    
   if(model == "lasso-cv"):
       SDH_AND_values = []
       YOUDEN_AND_values = []
       SDH_OR_values = []
       YOUDEN_OR_values = []
       
       # Initialise the progress bar
       pbar = tqdm(total=nsim, desc=f"Progress simple simulation {model} with nsample {nsample}")
       
       for i in range(nsim):
           if(simulation_method == "basic"):
              sample = simulation.generateSample_simBasic(nsample)
              neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_CV(sample, max_lambda)
               
              G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
              G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
               
              node_order =  ["X1", "X2", "X3", "X4", "X5"]
              known_structure = evaluation.create_known_structure_matrix_simpleSigmoid()
               
           elif(simulation_method == "smc-categories"):
              sample = simulation.generateSample_simSMC()
              neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_CV(sample, max_lambda)
              neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
               
              G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
              G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
               
              node_order = ['cance','histo','subco','stage','tpuri','pam50','menop', 'chemo']
              known_structure = evaluation.create_known_structure_matrix_SMCsim_categories()
            
           elif(simulation_method == "metabric-categories"):
             sample = simulation.generateSample_simMETABRIC()
             neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_CV(sample, max_lambda)
             neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
                
             G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
             G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
                
             node_order = ['Cance', 'Cellu', 'Pam50', 'Neopl', 'Histo', '3Gene', 'Tumor', 'Vital', 'Menop', 'Chemo', 'Hormo', 'Radio', 'PR', 'HER2', 'ER', 'Breas', 'Relap', 'sideR']
             known_structure = evaluation.create_known_structure_matrix_METABRICsim_categories()
               
           
           identified_structure_AND = evaluation.create_identified_structure_matrix(G_AND, node_order)
           SDH_AND_i, YOUDEN_AND_i = evaluation.evaluate_SDH_YOUDEN(known_structure, identified_structure_AND)
           SDH_AND_values.append(SDH_AND_i)
           YOUDEN_AND_values.append(YOUDEN_AND_i)
           
           identified_structure_OR = evaluation.create_identified_structure_matrix(G_OR, node_order)
           SDH_OR_i, YOUDEN_OR_i = evaluation.evaluate_SDH_YOUDEN(known_structure, identified_structure_OR)
           SDH_OR_values.append(SDH_OR_i)
           YOUDEN_OR_values.append(YOUDEN_OR_i)
           
           # Create a DataFrame for the current replication
           result_df = pd.DataFrame({
               'Model': [model] * nsim,
               'SimMethod': [simulation_method] * nsim,
               'Replication': list(range(nsim)),
               'Sample Size': [nsample] * nsim,
               'SDH (AND)': SDH_AND_i,
               'YOUDEN (AND)': YOUDEN_AND_i,
               'SDH (OR)': SDH_OR_i,
               'YOUDEN (OR)': YOUDEN_OR_i
           })
       
           # Append the result to the output file
           if not os.path.isfile(output_file) or i == 0:
               result_df.to_csv(output_file, mode='a', header=True, index=False)
           else:
               result_df.to_csv(output_file, mode='a', header=False, index=False)
       
           # Update progress bar
           pbar.update(1)
       
       # Close the progress bar
       pbar.close()
       
      
   elif(model == "bolasso-cv"):
       SDH_AND_values = []
       YOUDEN_AND_values = []
       SDH_OR_values = []
       YOUDEN_OR_values = []
       
       # Initialise the progress bar
       pbar = tqdm(total=nsim, desc=f"Progress simple simulation {model} with nsample {nsample}")
       
       for i in range(nsim):
           if(simulation_method == "basic"):
               sample = simulation.generateSample_simBasic(nsample)
               neighbourhoods = graphical_modelling.findNeighbourhoodsBOLASSO(sample, max_lambda)
               
               G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
               G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
               
               node_order =  ["X1", "X2", "X3", "X4", "X5"]
               known_structure = evaluation.create_known_structure_matrix_simpleSigmoid()
               
           elif(simulation_method == "smc-categories"):
               sample = simulation.generateSample_simSMC()
               neighbourhoods = graphical_modelling.findNeighbourhoodsBOLASSO(sample, max_lambda)
               neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
               
               G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
               G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
               
               node_order = ['cance','histo','subco','stage','tpuri','pam50','menop', 'chemo']
               known_structure = evaluation.create_known_structure_matrix_SMCsim_categories()
               
           elif(simulation_method == "metabric-categories"):
             sample = simulation.generateSample_simMETABRIC()
             neighbourhoods = graphical_modelling.findNeighbourhoodsBOLASSO(sample, max_lambda)
             neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
                
             G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
             G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
                
             node_order = ['Cance', 'Cellu', 'Pam50', 'Neopl', 'Histo', '3Gene', 'Tumor', 'Vital', 'Menop', 'Chemo', 'Hormo', 'Radio', 'PR', 'HER2', 'ER', 'Breas', 'Relap', 'sideR']
             known_structure = evaluation.create_known_structure_matrix_METABRICsim_categories()
               
           
           identified_structure_AND = evaluation.create_identified_structure_matrix(G_AND, node_order)
           SDH_AND_i, YOUDEN_AND_i = evaluation.evaluate_SDH_YOUDEN(known_structure, identified_structure_AND)
           SDH_AND_values.append(SDH_AND_i)
           YOUDEN_AND_values.append(YOUDEN_AND_i)
           
           identified_structure_OR = evaluation.create_identified_structure_matrix(G_OR, node_order)
           SDH_OR_i, YOUDEN_OR_i = evaluation.evaluate_SDH_YOUDEN(known_structure, identified_structure_OR)
           SDH_OR_values.append(SDH_OR_i)
           YOUDEN_OR_values.append(YOUDEN_OR_i)
           
           # Create a DataFrame for the current replication
           result_df = pd.DataFrame({
               'Model': [model] * nsim,
               'SimMethod': [simulation_method] * nsim,
               'Replication': list(range(nsim)),
               'Sample Size': [nsample] * nsim,
               'SDH (AND)': SDH_AND_i,
               'YOUDEN (AND)': YOUDEN_AND_i,
               'SDH (OR)': SDH_OR_i,
               'YOUDEN (OR)': YOUDEN_OR_i
           })
       
           # Append the result to the output file
           if not os.path.isfile(output_file) or i == 0:
               result_df.to_csv(output_file, mode='a', header=True, index=False)
           else:
               result_df.to_csv(output_file, mode='a', header=False, index=False)
       
           # Update progress bar
           pbar.update(1)
       
       # Close the progress bar
       pbar.close()
       
   elif(model == "lasso-aic"):
       SDH_AND_values = []
       YOUDEN_AND_values = []
       SDH_OR_values = []
       YOUDEN_OR_values = []
       
       # Initialise the progress bar
       pbar = tqdm(total=nsim, desc=f"Progress simple simulation {model} with nsample {nsample}")
       
       for i in range(nsim):
           if(simulation_method == "basic"):
               sample = simulation.generateSample_simBasic(nsample)
               neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_AIC(sample, max_lambda)
               
               G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
               G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
               
               node_order =  ["X1", "X2", "X3", "X4", "X5"]
               known_structure = evaluation.create_known_structure_matrix_simpleSigmoid()
               
           elif(simulation_method == "smc-categories"):
               sample = simulation.generateSample_simSMC()
               neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_AIC(sample, max_lambda)
               neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
               
               G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
               G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
               
               node_order = ['cance','histo','subco','stage','tpuri','pam50','menop', 'chemo']
               known_structure = evaluation.create_known_structure_matrix_SMCsim_categories()
               
           elif(simulation_method == "metabric-categories"):
             sample = simulation.generateSample_simMETABRIC()
             neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_AIC(sample, max_lambda)
             neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
                
             G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
             G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
                
             node_order = ['Cance', 'Cellu', 'Pam50', 'Neopl', 'Histo', '3Gene', 'Tumor', 'Vital', 'Menop', 'Chemo', 'Hormo', 'Radio', 'PR', 'HER2', 'ER', 'Breas', 'Relap', 'sideR']
             known_structure = evaluation.create_known_structure_matrix_METABRICsim_categories()
             
           
           identified_structure_AND = evaluation.create_identified_structure_matrix(G_AND, node_order)
           SDH_AND_i, YOUDEN_AND_i = evaluation.evaluate_SDH_YOUDEN(known_structure, identified_structure_AND)
           SDH_AND_values.append(SDH_AND_i)
           YOUDEN_AND_values.append(YOUDEN_AND_i)
           
           identified_structure_OR = evaluation.create_identified_structure_matrix(G_OR, node_order)
           SDH_OR_i, YOUDEN_OR_i = evaluation.evaluate_SDH_YOUDEN(known_structure, identified_structure_OR)
           SDH_OR_values.append(SDH_OR_i)
           YOUDEN_OR_values.append(YOUDEN_OR_i)
           
           # Create a DataFrame for the current replication
           result_df = pd.DataFrame({
               'Model': [model] * nsim,
               'SimMethod': [simulation_method] * nsim,
               'Replication': list(range(nsim)),
               'Sample Size': [nsample] * nsim,
               'SDH (AND)': SDH_AND_i,
               'YOUDEN (AND)': YOUDEN_AND_i,
               'SDH (OR)': SDH_OR_i,
               'YOUDEN (OR)': YOUDEN_OR_i
           })
       
           # Append the result to the output file
           if not os.path.isfile(output_file) or i == 0:
               result_df.to_csv(output_file, mode='a', header=True, index=False)
           else:
               result_df.to_csv(output_file, mode='a', header=False, index=False)
       
           # Update progress bar
           pbar.update(1)
       
       # Close the progress bar
       pbar.close()
       
   elif(model == "lasso-bic"):
       SDH_AND_values = []
       YOUDEN_AND_values = []
       SDH_OR_values = []
       YOUDEN_OR_values = []
       
       # Initialise the progress bar
       pbar = tqdm(total=nsim, desc=f"Progress simple simulation {model} with nsample {nsample}")
       
       for i in range(nsim):
           if(simulation_method == "basic"):
               sample = simulation.generateSample_simBasic(nsample)
               neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_BIC(sample, max_lambda)
               
               G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
               G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
               
               node_order =  ["X1", "X2", "X3", "X4", "X5"]
               known_structure = evaluation.create_known_structure_matrix_simpleSigmoid()
               
           elif(simulation_method == "smc-categories"):
               sample = simulation.generateSample_simSMC()
               neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_BIC(sample, max_lambda)
               neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
               
               G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
               G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
               
               node_order = ['cance','histo','subco','stage','tpuri','pam50','menop', 'chemo']
               known_structure = evaluation.create_known_structure_matrix_SMCsim_categories()  
           
           elif(simulation_method == "metabric-categories"):
              sample = simulation.generateSample_simMETABRIC()
              neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_BIC(sample, max_lambda)
              neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
                 
              G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
              G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
                 
              node_order = ['Cance', 'Cellu', 'Pam50', 'Neopl', 'Histo', '3Gene', 'Tumor', 'Vital', 'Menop', 'Chemo', 'Hormo', 'Radio', 'PR', 'HER2', 'ER', 'Breas', 'Relap', 'sideR']
              known_structure = evaluation.create_known_structure_matrix_METABRICsim_categories()
               
              
           identified_structure_AND = evaluation.create_identified_structure_matrix(G_AND, node_order)
           SDH_AND_i, YOUDEN_AND_i = evaluation.evaluate_SDH_YOUDEN(known_structure, identified_structure_AND)
           
           identified_structure_OR = evaluation.create_identified_structure_matrix(G_OR, node_order)
           SDH_OR_i, YOUDEN_OR_i = evaluation.evaluate_SDH_YOUDEN(known_structure, identified_structure_OR)

           # Create a DataFrame for the current iteration
           result_df = pd.DataFrame({
               'Model': [model],
               'SimMethod': [simulation_method],
               'Replication': [i],
               'Sample Size': [nsample],
               'SDH (AND)': [SDH_AND_i],
               'YOUDEN (AND)': [YOUDEN_AND_i],
               'SDH (OR)': [SDH_OR_i],
               'YOUDEN (OR)': [YOUDEN_OR_i]
           })
           # Append the result to the output file
           if not os.path.isfile(output_file) or i == 0:
               result_df.to_csv(output_file, mode='a', header=True, index=False)
           else:
               result_df.to_csv(output_file, mode='a', header=False, index=False)
       
           # Update progress bar
               pbar.update(1)
           
           # Close progress bar
           pbar.close()
           
       else:
           print("Method does not exist. Choose from: [lasso-cv, bolasso-cv, lasso-aic, lasso-bic]")  
           

def create_simulation_graph(dataset):
    """
    Creates graph from predefined conditional dependence structure of simulation

    Parameters:
    dataset (str): The name of the dataset ('smc' or 'metabric').

    Returns:
    G (networkx.Graph): The created simulation graph.
    """
    if dataset=='smc': 
        smc_sim = evaluation.create_known_structure_matrix_SMCsim_categories()
        
        smc_nodeorder = ['Cancertype', 'Histo',' Consens', 'TNMstage', 'Purity', 'Pam50',' Meno', 'Chemo']
        smc_binary = [' Meno', 'Chemo']
        smc_cat = ['Cancertype', 'Histo',' Consens', 'TNMstage', 'Purity', 'Pam50']
        
        G = graph_drawing.create_graph_from_structure_matrix(smc_sim, smc_nodeorder, smc_cat, smc_binary)
        return G
        
    elif dataset=='metabric':
        metabric_sim = evaluation.create_known_structure_matrix_METABRICsim_categories()
        
        metabric_nodeorder = ['Cance', 'Cellu', 'Pam50', 'Neopl', 'Histo', '3Gene', 'Tumor', 'Vital', 'Menop', 'Chemo', 'Hormo', 'Radio', 'PR', 'HER2', 'ER', 'Breas', 'Relap', 'sideR']
        metabric_cat = ['Cance', 'Cellu', 'Pam50', 'Neopl', 'Histo', '3Gene', 'Tumor', 'Vital']
        metabric_binary = ['Menop', 'Chemo', 'Hormo', 'Radio', 'PR', 'HER2', 'ER', 'Breas', 'Relap', 'sideR']
        
        G = graph_drawing.create_graph_from_structure_matrix(metabric_sim, metabric_nodeorder, categorical_vars=metabric_cat, binary_vars=metabric_binary)
    
    

def run_emperical_SMC():
    """
    Runs graphical modelling LASSO-AIC for SMC dataset

    Returns:
    smc_G (networkx.Graph): The constructed graph for the SMC dataset.
    """
    
    # Prepare emperica data
    smc_file = "data-external/BRCA-SMC-2018-clinical-data.tsv"
    smc_data = data_processing.load_data(smc_file)
    smc_data = data_processing.preprocess_data_SMC(smc_data)
    smc_binary = ['Cancer Type Detailed_Breast Invasive Ductal Carcinoma',
     'Histology _Ductal Carcinoma', 'TNM Stage_1A', 'TNM Stage_2A',
     'TNM Stage_2B', 'TNM Stage_3A', 'TNM Stage_3C',
     'Menopausal Status At Diagnosis_Post',
     'Menopausal Status At Diagnosis_Pre', 'PAM50 subtype_Basal',
     'PAM50 subtype_Her2', 'PAM50 subtype_LuminalA',
     'PAM50 subtype_LuminalB', 'Subtype Consensus_ER+',
     'Subtype Consensus_ER+HER2+', 'Subtype Consensus_HER2+',
     'Subtype Consensus_TN', 'Chemotherapy_No', 'Chemotherapy_Yes']
    smc_data.columns = smc_data.columns.str.replace(' ', '_')
    smc_binary = [col.replace(' ', '_') for col in smc_binary]
    
    # Apply graphical modelling 
    smc_neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_AIC_external(smc_data, smc_binary, max_lambda=200)
    smc_neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(smc_neighbourhoods)

    # Construct graph
    smc_G = graph_drawing.constructGraph_AND(smc_neighbourhoods)
    smc_G_or = graph_drawing.constructGraph_OR(smc_neighbourhoods)
    
    return smc_G


def run_emperical_METABRIC():
    """
    Runs graphical modelling LASSO-BIC for METABRIC dataset

    Returns:
    metabric_G (networkx.Graph): The constructed graph for the METABRIC dataset.
    """
    
    # Prepare empirical data
    metabric_file = "data-external/BRCA-METABRIC-2012-2016-clinical-data.tsv"
    metabric_data = data_processing.load_data(metabric_file)
    metabric_data = data_processing.preprocess_data_METABRIC(metabric_data)
    metabric_binary = ['Type of Breast Surgery_BREAST CONSERVING',
    'Type of Breast Surgery_MASTECTOMY',
    'Cancer Type Detailed_Breast Invasive Ductal Carcinoma',
    'Cancer Type Detailed_Breast Invasive Lobular Carcinoma',
    'Cancer Type Detailed_Breast Invasive Mixed Mucinous Carcinoma',
    'Cancer Type Detailed_Breast Mixed Ductal and Lobular Carcinoma',
    'Cellularity_High', 'Cellularity_Low', 'Cellularity_Moderate',
    'Chemotherapy_NO', 'Chemotherapy_YES',
    'Pam50 + Claudin-low subtype_Basal', 'Pam50 + Claudin-low subtype_Her2',
    'Pam50 + Claudin-low subtype_LumA', 'Pam50 + Claudin-low subtype_LumB',
    'Pam50 + Claudin-low subtype_Normal',
    'Pam50 + Claudin-low subtype_claudin-low', 'ER Status_Negative',
    'ER Status_Positive', 'Neoplasm Histologic Grade_1.0',
    'Neoplasm Histologic Grade_2.0', 'Neoplasm Histologic Grade_3.0',
    'HER2 Status_Negative', 'HER2 Status_Positive',
    'Tumor Other Histologic Subtype_Ductal/NST',
    'Tumor Other Histologic Subtype_Lobular',
    'Tumor Other Histologic Subtype_Medullary',
    'Tumor Other Histologic Subtype_Mixed',
    'Tumor Other Histologic Subtype_Mucinous',
    'Tumor Other Histologic Subtype_Tubular/ cribriform',
    'Hormone Therapy_NO', 'Hormone Therapy_YES',
    'Inferred Menopausal State_Post', 'Inferred Menopausal State_Pre',
    'Primary Tumor Laterality_Left', 'Primary Tumor Laterality_Right',
    'Overall Survival Status_0:LIVING',
    'Overall Survival Status_1:DECEASED', 'PR Status_Negative',
    'PR Status_Positive', 'Radio Therapy_NO', 'Radio Therapy_YES',
    'Relapse Free Status_0:Not Recurred', 'Relapse Free Status_1:Recurred',
    '3-Gene classifier subtype_ER+/HER2- High Prolif',
    '3-Gene classifier subtype_ER+/HER2- Low Prolif',
    '3-Gene classifier subtype_ER-/HER2-',
    '3-Gene classifier subtype_HER2+', 'Tumor Stage_1.0', 'Tumor Stage_2.0',
    'Tumor Stage_3.0', 'Tumor Stage_4.0',
    'Patient\'s Vital Status_Died of Disease',
    'Patient\'s Vital Status_Died of Other Causes',
    'Patient\'s Vital Status_Living']
    metabric_data.columns = metabric_data.columns.str.replace(' ', '_').str.replace("'", '')
    metabric_binary = [col.replace(' ', '_').replace("'", '') for col in metabric_binary]
    
    # Apply graphical modelling 
    metabric_neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_BIC_external(metabric_data, metabric_binary, max_lambda=200)
    metabric_neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(metabric_neighbourhoods)
    
    # Construct graph
    metabric_G = graph_drawing.constructGraph_AND(metabric_neighbourhoods)
    
    return metabric_G
     
     
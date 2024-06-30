"""
Author: Jens van Drunen (572793fd)
Affiliation: Erasmus University Rotterdam
Email: 572793fd@eur.nl  
Description: simulation.py - Cosnsists of all functionalities regarding the (1) basic, (2) SMC and (3) METABRIC simulations
"""
import logging
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

def sigmoid(x):
    """
    Computes the sigmoid function.

    Parameters:
    x (float or np.array): Input value or array.

    Returns:
    float or np.array: Sigmoid of the input.
    """
    return 1 / (1 + np.exp(-x))


def generateSample_simBasic(n=1000):
    """
    Generates basic simulation

    Parameters:
    n (int): Number of samples to generate. Default is 1000.

    Returns:
    pd.DataFrame: Generated dataset with columns ["X1", "X2", "X3", "X4", "X5"].
    """
    
    # Initialise the data array
    data = np.zeros((n, 5))
    
    # X1: Independent variable
    data[:, 0] = np.random.binomial(1, 0.5, n)
    
    # X2 depends on X1
    logits_X2 = -3 * data[:, 0]
    prob_X2 = sigmoid(logits_X2)
    data[:, 1] = np.random.binomial(1, prob_X2)
    
    # X3 depends on X1
    logits_X3 = 3 * data[:, 0]
    prob_X3 = sigmoid(logits_X3)
    data[:, 2] = np.random.binomial(1, prob_X3)
    
    # X4 depends on X2 and X3
    logits_X4 = 1 + 2 * data[:, 1] - 2 * data[:, 2]
    prob_X4 = sigmoid(logits_X4)
    data[:, 3] = np.random.binomial(1, prob_X4)
    
    # X5: independent variable
    data[:, 4] = np.random.binomial(1, 0.5, n)
    
    return pd.DataFrame(data=data, columns=["X1","X2","X3","X4","X5"])


def generateSample_simSMC(n=187):
    """
    Generates a simulated dataset resembling emperical SMC data ewith multiple categorical and binary variables.

    Parameters:
    n (int): Number of samples to generate. Default is 187 (based on number of observations SMC)

    Returns:
    pd.DataFrame: Generated dataset with various categorical and binary variables representing cancer-related data.
    """
    
    # Initialise data arrays
    data_cancertype = np.zeros((n, 3))
    data_hist = np.zeros((n, 3))
    data_pam50 = np.zeros((n, 5))
    data_subconsens = np.zeros((n, 4))
    data_tpurity = np.zeros((n, 3))
    data_stage = np.zeros((n, 3))
    
    # Cat1: Cancertype (3 variables)
    p_cancertype = [0.08, 0.84, 0.08]
    for i in range(n):
        category = np.argmax(np.random.multinomial(1, p_cancertype))
        data_cancertype[i, category] = 1
    
    # Cat2: Histology (3 variables) -- Conditional dependent on cancer type
    for i in range(n):
        if data_cancertype[i, 1] == 1:  # if cancer type 2 is active
            logits_hist = np.array([3.0, -2.0, -2.0])
        elif data_cancertype[i, 0] == 1:
            logits_hist = np.array([-1.0, 3.0, -1.0])
        else:
            logits_hist = np.array([-1.0, -1.0, 3.0])
        
        # Normalise probabilities with softmax function
        prob_hist = np.exp(logits_hist)
        prob_hist /= prob_hist.sum()  
        
        category = np.argmax(np.random.multinomial(1, prob_hist))
        data_hist[i, category] = 1
        
    # Cat 3: Immunohistochemisty Subtype Consensus (4 variables)
    p_subconsens = [0.55, 0.15, 0.09, 0.21]
    for i in range(n):
        category = np.argmax(np.random.multinomial(1, p_subconsens))
        data_subconsens[i, category] = 1
        
    # Cat 4: TNM Stage (3 variables)
    p_stage = [0.16, 0.54, 0.30]
    for i in range(n):
        category = np.argmax(np.random.multinomial(1, p_stage))
        data_stage[i, category] = 1
    
    # Cat 5: Tumor Purity (3 variables) -- Conditional dependent on TNM stage
    for i in range(n):
        if data_stage[i, 0] == 1:
            logits_tpurity = np.array([1.5, 0.0, -1.5])
        elif data_stage[i, 1] == 1:
            logits_tpurity = np.array([-1.0, 2.0, 1.0])
        else:
            logits_tpurity = np.array([-2.0, 2.0, 0.2])
        
        # Normalise probabilities with softmax function
        prob_tpurity = np.exp(logits_tpurity)
        prob_tpurity /= prob_tpurity.sum()  
        
        category = np.argmax(np.random.multinomial(1, prob_tpurity))
        data_tpurity[i, category] = 1
        
    # Cat 5: Pam50 Subtype (5 variables) -- Conditional dependent on subtype consensus
    p_pam50_base = np.array([0.19, 0.10, 0.25, 0.40, 0.06])
    for i in range(n):
        if data_subconsens[i, 1] == 1 or data_subconsens[i, 2] == 1:
            logits_pam50 = np.log(p_pam50_base) + np.array([0.0, 2.0, 0.0, 0.0, 0.0])
        else:
            logits_pam50 = np.log(p_pam50_base)
        
        # Normalise probabilities with softmax function
        prob_pam50 = np.exp(logits_pam50)
        prob_pam50 /= prob_pam50.sum()  # Normalize
        category = np.argmax(np.random.multinomial(1, prob_pam50))
        data_pam50[i, :] = 0
        data_pam50[i, category] = 1
    
    # BIN1  Menopausal
    p_menopausal_pre = 0.88
    menopausal = np.random.binomial(1, p_menopausal_pre, n)
    
    # BIN2: Chemotherapy 
    logits_chemo = -0.1 - 1.6 * data_subconsens[:, 3] + 1.7 * menopausal
    prob_chemo = sigmoid(logits_chemo)
    chemo = np.random.binomial(1, prob_chemo, n)
    
    # Combine all data into a DataFrame
    data = np.hstack([data_cancertype, data_hist, data_pam50, data_subconsens, data_tpurity, data_stage, menopausal.reshape(-1, 1), chemo.reshape(-1, 1)])
    columns = ['cancertype_1', 'cancertype_2', 'cancertype_3',
               'histo_1', 'histo_2', 'histo_3',
               'subconsens_1', 'subconsens_2', 'subconsens_3', 'subconsens_4',
               'stage_1', 'stage_2', 'stage_3',
               'tpurity_1', 'tpurity_2', 'tpurity_3',
               'pam50_1', 'pam50_2', 'pam50_3', 'pam50_4', 'pam50_5',
               'menopausal', 'chemo']
    
    df = pd.DataFrame(data, columns=columns)
    return df
      

def generateSample_simMETABRIC(n=1096):
    """
    Generates a simulated dataset resembling emperical SMC data ewith multiple categorical and binary variables.

    Parameters:
    n (int): Number of samples to generate. Default is 1096 (based on number of observations METABRIC data)

    Returns:
    pd.DataFrame: Generated dataset with various categorical and binary variables representing METABRIC data.
    """
    
    # Independent variables (probabilties based on occurences in METABRIC)
    
    # BIN1: Tumor side 
    tumor_right = np.random.binomial(1, 0.48, n)
    
    # BIN2:  menopausal state pre 
    menopausal_pre = np.random.binomial(1, 0.22, n)
    
    # BIN3: PR status positive
    PR = np.random.binomial(1, 0.41, n)
    
    # BIN4: HER2 status positive
    HER2 = np.random.binomial(1, 0.72, n)
    
    # BIN5: ER status positive
    ER = np.random.binomial(1, 0.72, n)
    

    # Categorical variables (probabilties based on occurences in METABRIC)
    
    # CAT1: Cancer Type (5 variables)
    p_cat_cancertype = [0.789, 0.128, 0.07, 0.013]
    cat_data_cancertype = np.zeros((n, 4))
    for i in range(n):
        category = np.argmax(np.random.multinomial(1, p_cat_cancertype))
        cat_data_cancertype[i, category] = 1

    # CAT2: Tumor Other Histologic Subtype (5 variables) dependent on cancer type
    p_cat_histologicsub = [0.75, 0.07, 0.01, 0.05, 0.11]
    cat_data_histologicsub = np.zeros((n, 5))
    for i in range(n):
        if cat_data_cancertype[i,0] == 1:
            histo_logits = np.array([2, 0.3, 0.2, 0, 0])
        elif cat_data_cancertype[i,1] ==1:
            histo_logits = np.array([0.3, 2.5, 0.1, 0, 0])
        elif cat_data_cancertype[i,2] ==1:
            histo_logits = np.array([0.5, 0, 3.5, 0.4, 0])
        elif cat_data_cancertype[i,3] ==1:
            histo_logits = np.array([0.5, 0 , 0, 2.0, 2.5])
            
        # Normalise probabilities with softmax function
        prob_histologicsub = np.exp(histo_logits)
        prob_histologicsub /= np.sum(prob_histologicsub)
        
        category = np.argmax(np.random.multinomial(1, prob_histologicsub))
        cat_data_histologicsub[i, category] = 1

    # CAT3: Neoplasm Histologic Grade (3 variables)
    p_cat_neoplasm = [0.39, 0.49, 0.12]
    cat_data_neoplasm = np.zeros((n, 3))
    for i in range(n):
        category = np.argmax(np.random.multinomial(1, p_cat_neoplasm))
        cat_data_neoplasm[i, category] = 1

    # CAT4: 3-Gene classifier subtype (4 variables)
    p_cat_3gene = [0.32, 0.43, 0.15, 0.10]
    cat_data_3gene = np.zeros((n, 4))
    for i in range(n):
        if ER[i]==1 and HER2[i]==1:
            p_cat_3gene = [0.05, 0.05, 0.90, 0.0] 
        category = np.argmax(np.random.multinomial(1, p_cat_3gene))
        cat_data_3gene[i, category] = 1
        
    # CAT5: Cellularity Low, Mid, High (3 variables) conditional dependent on neoplasm lvl
    cat_data_cellularity = np.zeros((n, 3))
    for i in range(n):
        if cat_data_neoplasm[i, 0] == 1:
            logits_tumorstage = np.array([1.5, 1, -2])
        elif cat_data_neoplasm[i, 1] == 1:
            logits_tumorstage = np.array([0.5, 1.5, -1.5])
        elif cat_data_neoplasm[i, 2] == 1:
            logits_tumorstage = np.array([-1, 1, 2.0])

        # Normalise probabilities with softmax function
        prob_cellularity = np.exp(logits_tumorstage)
        prob_cellularity /= np.sum(prob_cellularity)
        
        category = np.argmax(np.random.multinomial(1, prob_cellularity))
        cat_data_cellularity[i, category] = 1
        
    # CAT6: Tumor stage (3 variables) CONDITIONAL DEPENDENT ON Neoplasm lvl
    cat_data_tumorstage = np.zeros((n, 3))
    p_cat_tumorstage = [0.25, 0.50, 0.25]
    for i in range(n):
        if cat_data_neoplasm[i, 0] == 1:
            logits_tumorstage = np.array([2, -1, -1.5])
        elif cat_data_neoplasm[i, 1] == 1:
            logits_tumorstage = np.array([0.0, 1.0, -1.5])
        elif cat_data_neoplasm[i, 2] == 1:
            logits_tumorstage = np.array([-2.0, 1, 2.0])
            
        # Normalise probabilities with softmax function
        prob_tumorstage = np.exp(logits_tumorstage)
        prob_tumorstage /= np.sum(prob_tumorstage)
        
        category = np.argmax(np.random.multinomial(1, prob_tumorstage))
        cat_data_tumorstage[i, category] = 1
        
    # CAT7: Pam50 Claudin-low subtype (6 variables)
    p_cat_pam50 = [0.12, 0.37, 0.24, 0.07, 0.09, 0.11]
    cat_data_pam50 = np.zeros((n, 6))
    for i in range(n):
        if cat_data_neoplasm[i,2] == 1:
            p_cat_pam50 = [0.90, 0.05, 0.05, 0, 0, 0]
        category = np.argmax(np.random.multinomial(1, p_cat_pam50))
        cat_data_pam50[i, category] = 1

    # CAT8: Vital Status Patient (3 variables)
    p_cat_vital = [0.41, 0.34, 0.25]
    cat_data_vital = np.zeros((n, 3))
    for i in range(n):
        category = np.argmax(np.random.multinomial(1, p_cat_vital))
        cat_data_vital[i, category] = 1

   
    # BIN6: Receive chemo therapy
    logits_chemo = 2.0 + 1.2 * cat_data_tumorstage[:,2] - 1.2 * cat_data_tumorstage[:,0] - 1.8 * HER2 - 1.8 * ER
    prob_chemo = sigmoid(logits_chemo)
    chemo = np.random.binomial(1, prob_chemo, n)

    # BIN7: Receive hormone therapy
    logits_hormone = -1 + 1.4 * menopausal_pre + 1.5 * ER + 1.5 * HER2   
    prob_hormone = sigmoid(logits_hormone)
    hormone = np.random.binomial(1, prob_hormone, n)

    # BIN8: Receive radio therapy
    logits_radio = 0.5 + 1.5 * PR - 3 * HER2 
    prob_radio = sigmoid(logits_radio)
    radio = np.random.binomial(1, prob_radio, n)

    # BIN9: Receive breast surgery
    logits_surgery = 2.0 * cat_data_tumorstage[:,2] - 1.5 * cat_data_tumorstage[:,0] 
    prob_breast_surgery = sigmoid(logits_surgery)
    breast_surgery = np.random.binomial(1, prob_breast_surgery, n)

    # BIN10: Relapse free
    logits_relapseFree = -10 * cat_data_vital[:, 1] + 2 * cat_data_tumorstage[:,0] - 2 * cat_data_tumorstage[:,2]
    prob_relapseFree = sigmoid(logits_relapseFree)
    relapseFree = np.random.binomial(1, prob_relapseFree, n)

    # Combine data into a DataFrame
    data = np.hstack([
        cat_data_cancertype,
        cat_data_cellularity,
        cat_data_pam50,
        cat_data_neoplasm,
        cat_data_histologicsub,
        cat_data_3gene,
        cat_data_tumorstage,
        cat_data_vital,
        menopausal_pre[:, np.newaxis],
        chemo[:, np.newaxis],
        hormone[:, np.newaxis],
        radio[:, np.newaxis],
        PR[:, np.newaxis],
        HER2[:, np.newaxis],
        ER[:, np.newaxis],
        breast_surgery[:, np.newaxis],
        relapseFree[:, np.newaxis],
        tumor_right[:, np.newaxis]
    ])

    columns = [
        'CancerType_1', 'CancerType_2', 'CancerType_3', 'CancerType_4',
        'Cellularity_Low', 'Cellularity_Mid', 'Cellularity_High',
        'Pam50_1', 'Pam50_2', 'Pam50_3', 'Pam50_4', 'Pam50_5', 'Pam50_6',
        'Neoplasm_1', 'Neoplasm_2', 'Neoplasm_3',
        'HistologicSub_1', 'HistologicSub_2', 'HistologicSub_3', 'HistologicSub_4', 'HistologicSub_5',
        '3Gene_1', '3Gene_2', '3Gene_3', '3Gene_4',
        'TumorStage_1', 'TumorStage_2', 'TumorStage_3',
        'VitalStatus_1', 'VitalStatus_2', 'VitalStatus_3', 'MenopausalPre',
        'Chemo', 'Hormone', 'Radio', 'PR', 'HER2', 'ER', 'BreastSurgery', 'RelapseFree', 'sideRight'
    ]

    return pd.DataFrame(data, columns=columns)
  
    


    
    


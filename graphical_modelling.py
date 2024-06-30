"""
Author: Jens van Drunen (572793fd)
Affiliation: Erasmus University Rotterdam
Email: 572793fd@eur.nl
Description: graphical_modelling.py - Contains functionalities for LASSO-based graphical modelling.
"""
import logging
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import log_loss
from tqdm import tqdm

# Initialise logger
logger = logging.getLogger(__name__)

def calculate_AIC(X, y, model):
    """
    Calculates the AIC for a given (BO)LASSO model.

    Parameters:
    X (pd.DataFrame): Predictors of the model.
    y (pd.Series): Outcome variable.
    model (LogisticRegression): Fitted logistic regression model.

    Returns:
    AIC (float): AIC value.
    """
    
    # Number of parameters (non-zero coefficients + intercept)
    k = np.sum(model.coef_ != 0) + 1
    
    # Log likelihood (note: log_loss returns the negative log-likelihood)
    log_likelihood = -log_loss(y, model.predict_proba(X)[:, 1], normalize=False)
    
    # AIC calculation
    AIC = 2 * k - 2 * log_likelihood

    return AIC

def calculate_BIC(X, y, model):
    """
    Calculates the BIC for a given (BO)LASSO model.

    Parameters:
    X (pd.DataFrame): Predictors of the model.
    y (pd.Series): Outcome variable.
    model (LogisticRegression): Fitted logistic regression model.

    Returns:
    BIC (float): BIC value.
    """
    
    # Number of observations
    n = len(y)
    
    # Number of parameters (non-zero coefficients + intercept)
    k = np.sum(model.coef_ != 0) + 1

    # Log likelihood (note: log_loss returns the negative log-likelihood)
    log_likelihood = -log_loss(y, model.predict_proba(X)[:, 1], normalize=False)
    
    # BIC calculation
    BIC = k * np.log(n) - 2 * log_likelihood

    return BIC
    
def findNeighbourhoodsLASSO_CV(data, max_lambda=5):
    """
    Finds the neighbourhoods using LASSO with Cross-Validation (CV) for model selection.

    Parameters:
    data (pd.DataFrame): Binary dataset.
    max_lambda (float): The maximum lambda value to consider.

    Returns:
    neighbourhoods (dict): Dictionary of neighbourhoods for each variable.
    """
    
    # Initialise search space 
    min_C = 1/max_lambda
    Cs = np.logspace(np.log10(min_C), np.log10(1), 100)

    # Regress each variable on all other variables
    neighbourhoods = {}
    for col in data.columns:
        y = data[col]
        X = data.drop(columns=[col])
        
        # Skip columns with only one unique value
        if len(y.unique()) <= 1:
            neighbourhoods[col] = []
            logger.info(f"column {col} skipped because it contains only 1 value")
            continue

        lasso = LogisticRegressionCV(Cs=Cs, cv=5, penalty='l1', solver='saga', max_iter=10000)
        lasso.fit(X, y)

        # Find non-zero regression coefficients to construct neighbourhood col
        non_zero_coefs = np.where(lasso.coef_[0] != 0)[0]
        neighbours = X.columns[non_zero_coefs]
        neighbourhoods[col] = neighbours

    return neighbourhoods

def findNeighbourhoodsLASSO_AIC(data, max_lambda=5):
    """
    Finds the neighbourhoods using LASSO with AIC for model selection.

    Parameters:
    data (pd.DataFrame): Binary dataset.
    max_lambda (float): The maximum lambda value to consider.

    Returns:
    neighbourhoods (dict): Dictionary of neighbourhoods for each variable.
    """
    
    # Initialise search space 
    min_C = 1/max_lambda
    Cs = np.logspace(np.log10(min_C), np.log10(1), 100)
    
    # Regress each variable on all other variables
    neighbourhoods = {}
    for col in data.columns:
        y = data[col]
        X = data.drop(columns=[col])
        
        # Skip columns with only one unique value
        if len(y.unique()) <= 1:
            neighbourhoods[col] = []
            logger.info(f"column {col} skipped because it contains only 1 value")
            continue

        best_aic = 100000
        best_model = None
        best_C = None

        for C in Cs:
            lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, C=C)
            lasso.fit(X, y)

            aic = calculate_AIC(X,y,lasso)
            if aic < best_aic:
                best_aic = aic
                best_model = lasso
                best_C = C

        # Find non-zero regression coefficients to construct neighbourhood
        non_zero_coefs = np.where(best_model.coef_[0] != 0)[0]
        neighbours = X.columns[non_zero_coefs]
        neighbourhoods[col] = neighbours

    return neighbourhoods

def findNeighbourhoodsLASSO_BIC(data, max_lambda=5):
    """
    Finds the neighbousrhoods using LASSO with BIC for model selection.

    Parameters:
    data (pd.DataFrame): Binary dataset.
    max_lambda (float): The maximum lambda value to consider.

    Returns:
    neighbourhoods (dict): Dictionary of neighbourhoods for each variable.
    """
    
    # Create range of lambda's to search
    min_C = 1/max_lambda
    Cs = np.logspace(np.log10(min_C), np.log10(1), 100)
    
    # Regress each variable on all other variables
    neighbourhoods = {}
    for col in data.columns:
        y = data[col]
        X = data.drop(columns=[col])
        
        # Skip columns with only one unique value
        if len(y.unique()) <= 1:
            neighbourhoods[col] = []
            logger.info(f"column {col} skipped because it contains only 1 value")
            continue

        best_bic = 100000
        best_model = None
        best_C = None

        for C in Cs:
            lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, C=C)
            lasso.fit(X, y)
            
            bic = calculate_BIC(X,y,lasso)
            
            if bic < best_bic:
                best_bic = bic
                best_model = lasso
                best_C = C
        
        # Find non-zero regression coefficients to construct neighbourhood
        non_zero_coefs = np.where(best_model.coef_[0] != 0)[0]
        neighbours = X.columns[non_zero_coefs]

        neighbourhoods[col] = neighbours

    return neighbourhoods


def findNeighbourhoodsBOLASSO(data, B=40, pi_cut=0.90, max_lambda=5):
    """
    Finds the neighbourhoods using Bolasso (Bootstrap Lasso).

    Parameters:
    data (pd.DataFrame): Binary dataset.
    B (int): Number of bootstrap samples.
    max_lambda (float): The maximum lambda value to consider.
    pi_cut (float): Cut-off value for the definition of neighbourhood.

    Returns:
    neighbourhoods (dict): Dictionary of neighbourhoods for each variable.
    """
    
    # Initialise search space 
    min_C = 1/max_lambda
    Cs = np.logspace(np.log10(min_C), np.log10(1), 100)
    
    # Initialise presence matrix
    n, p = data.shape
    presence_count = {col: {col: 0 for col in data.columns} for col in data.columns}
    
    # Generate bootstrap samples
    samples =  [resample(data) for i in range(B)]
    
    # Regress each variable on all other variables
    neighbourhoods = {}
    for sample in samples:
        for col in data.columns:
            y = sample[col]
            X = sample.drop(columns=col)
            
            # Skip columns with only one unique value
            if len(y.unique()) <= 1:
                neighbourhoods[col] = []
                logger.info(f"column {col} skipped because it contains only 1 value")
                continue
            
            lasso = LogisticRegressionCV(Cs=Cs, cv=5, penalty='l1', solver='saga', max_iter=10000)
            lasso.fit(X,y)
            
            optimal_lambda = 1 / lasso.C_[0]
            
            # Find non-zero regression coefficients to construct neighbourhood
            non_zero_coefs = np.where(lasso.coef_[0] != 0)[0] 
            neighbours = X.columns[non_zero_coefs]
            
            # Update presence matrix
            for neighbour in neighbours:
                presence_count[col][neighbour] += 1
          
    # Construct final neighbourhoods based on presence count 
    neighbourhoods = {col: [] for col in data.columns}
    for col in data.columns:
        for neighbour in presence_count[col]:
            if presence_count[col][neighbour] / B >= pi_cut:
                neighbourhoods[col].append(neighbour)
                
    return neighbourhoods


def findNeighbourhoodsLASSO_CV_external(data, binary_variables, max_lambda=5):
    """
    Finds the neighbourhoods using LASSO with Cross-Validation (CV) for model selection

    Parameters:
    data (pd.DataFrame): Binary dataset
    max_lambda (float): The maximum lambda value to consider
    binary_variables (list): Variables to include in graphical modelling

    Returns:
    neighbourhoods (dict): Dictionary of neighbourhoods for each variable
    """
    
    # Initialise search space
    min_C = 1/max_lambda
    Cs = np.logspace(np.log10(min_C), np.log10(1), 100)

    # Regress each variable on all other variables
    neighbourhoods = {}
    for col in binary_variables:
        logger.info(f"Start LASSO for col {col}")
        
        y = data[col]
        X = data.drop(columns=[col])

        lasso = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=50000)
        lasso.fit(X, y)

        # Find non-zero regression coefficients to construct neighbourhood
        non_zero_coefs = np.where(lasso.coef_[0] != 0)[0]
        all_neighbours = X.columns[non_zero_coefs]
        
        # Filter to keep only binary variables as neighbours
        neighbours = [neighbour for neighbour in all_neighbours if neighbour in binary_variables]
        neighbourhoods[col] = neighbours

    return neighbourhoods

def findNeighbourhoodsLASSO_AIC_external(data, binary_variables, max_lambda=5):
    """
    Finds the neighbourhoods using LASSO with AIC for model selection.

    Parameters:
    data (pd.DataFrame): Binary dataset.
    max_lambda (float): The maximum lambda value to consider.
    binary_variables (list): Variables to include in graphical modelling.

    Returns:
    neighbourhoods (dict): Dictionary of neighbourhoods for each variable.
    """
    
    # Initialise search space
    min_C = 1/max_lambda
    Cs = np.logspace(np.log10(min_C), np.log10(1), 1000)

    # Initialise progress bar 
    progress_bar = tqdm(total=len(binary_variables), desc="Processing variables")    

    # Regress each variable on all other variables
    neighbourhoods = {}
    for col in binary_variables:
        
        y = data[col]
        X = data.drop(columns=[col])
        
        # Skip columns with only one unique value 
        if len(y.unique()) <= 1:
            neighbourhoods[col] = []
            logger.info(f"column {col} skipped because it contains only 1 value")
            continue

        best_aic = 100000
        best_model = None
        
        # Search through lambda search-space to find optimal penalty based on AIC
        progress_bar_inside = tqdm(total=len(Cs), desc="Finding best model")
        for C in Cs:
            lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=30000, C=C)
            lasso.fit(X, y)
            
            aic = calculate_AIC(X,y,lasso)
            
            if aic < best_aic:
                best_aic = aic
                best_model = lasso
                
            progress_bar_inside.update()
            
        progress_bar_inside.close()

        # Find non-zero regression coefficients to construct neighbourhood
        non_zero_coefs = np.where(best_model.coef_[0] != 0)[0]
        neighbours = X.columns[non_zero_coefs]
        
        # Filter neighbours to include only binary variables
        binary_neighbours = [neighbour for neighbour in neighbours if neighbour in binary_variables]
        neighbourhoods[col] = binary_neighbours
        
        progress_bar.update(1)

    progress_bar.close()
    
    return neighbourhoods

def findNeighbourhoodsLASSO_BIC_external(data, binary_variables, max_lambda=5):
    """
    Finds the neighbourhoods using LASSO with BIC for model selection.

    Parameters:
    data (pd.DataFrame): Binary dataset.
    max_lambda (float): The maximum lambda value to consider.
    binary_variables (list): Variables to include in graphical modelling.

    Returns:
    neighbourhoods (dict): Dictionary of neighbourhoods for each variable.
    """
    
    # Initialise search space
    min_C = 1/max_lambda
    Cs = np.logspace(np.log10(min_C), np.log10(1), 100)
    
    # Initialise progress bar 
    progress_bar = tqdm(total=len(binary_variables), desc="Processing variables")
    
    # Regress each variable on all other variables
    neighbourhoods = {}
    for col in binary_variables:
        
        y = data[col]
        X = data.drop(columns=[col])
        
        # Skip columns with only one unique value
        if len(y.unique()) <= 1:
            neighbourhoods[col] = []
            logger.info(f"column {col} skipped because it contains only 1 value")
            continue

        best_bic = 100000
        best_model = None
        
        progress_bar_inside = tqdm(total=len(Cs), desc="Finding best model")
        
        # Search through lambda search-space to find optimal penalty based on BIC
        for C in Cs:
            lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=30000, C=C)
            lasso.fit(X, y)
            
            bic = calculate_BIC(X,y,lasso)
            
            if bic < best_bic:
                best_bic = bic
                best_model = lasso
                
            progress_bar_inside.update()
            
        progress_bar_inside.close()
        
        # Find non-zero regression coefficients to construct neighbourhood
        non_zero_coefs = np.where(best_model.coef_[0] != 0)[0]
        neighbours = X.columns[non_zero_coefs]
        
        # Filter neighbours to include only binary variables
        binary_neighbours = [neighbour for neighbour in neighbours if neighbour in binary_variables]
        neighbourhoods[col] = binary_neighbours
        
        progress_bar.update(1)

    progress_bar.close()
    
    return neighbourhoods


def generaliseNeigbourhoodsToCategories(neighbourhoods):
    """
    Generalises neighbourhoods through transforming dummy variables back to categories, based on variable name

    Parameters:
    neighbourhoods (dict): Dictionary of neighbourhoods for each variable.

    Returns:
    dict: Generalized neighbourhoods by categories.
    """
    
    def get_category(variable):
        # Define robust way to extract categories by splitting by underscores
        parts = variable.split('_')
        return parts[0] if parts else variable
    
    category_neighbourhoods = {}

    # Initialise categories
    for variable in neighbourhoods:
        category = get_category(variable)
        if category not in category_neighbourhoods:
            category_neighbourhoods[category] = set()

    # Collect neighbours for each category
    for variable, neighbours in neighbourhoods.items():
        variable_category = get_category(variable)
        for neighbour in neighbours:
            neighbour_category = get_category(neighbour)
            if variable_category != neighbour_category:
                category_neighbourhoods[variable_category].add(neighbour_category)

    return category_neighbourhoods
    
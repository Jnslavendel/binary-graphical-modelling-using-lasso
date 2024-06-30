"""
Author: Jens van Drunen (572793fd)
Affiliation: Erasmus University Rotterdam
Email: 572793fd@eur.nl
Description: 
"""
import simulation 
import data_processing
import graphical_modelling
import graph_drawing
import evaluation 
import numpy as np
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import usage

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def main():
    
    usage.run_emperical_SMC()
    
if __name__ == "__main__":
    main()
# Graphical Modelling Using LASSO

## Overview

This project implements graphical modelling using LASSO (Least Absolute Shrinkage and Selection Operator) for both simulated and empirical datasets. The scripts includes data simulation, processing, graphical model construction, and evaluation. 

## Author

- **Name**: Jens van Drunen
- **Email**: 572793fd@eur.nl

## Project Structure

- `main.py`: Entry point of the project
- `simulation.py`: Contains functions for generating simulated datasets.
- `data_processing.py`: Functions for processing empirical datasets (SMC and METABRIC).
- `graphical_modelling.py`: LASSO-based graphical modelling functionalities.
- `graph_drawing.py`: Functions for constructing and visualising graphs.
- `evaluation.py`: Functions for evaluating the performance of graphical models.
- `usage.py`: Integrates various functionalities to generate concrete results.

## Project Setup

### Prerequisites

- Python 3.6+
- Required packages (install via `pip`):
  - numpy
  - pandas
  - scikit-learn
  - networkx
  - matplotlib
  - seaborn
  - tqdm
  - openpyxl 

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/graphical-modelling-lasso.git](https://github.com/Jnslavendel/binary-graphical-modelling-using-lasso.git
    cd graphical-modelling-lasso
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage 

#### Generating Simulation Data 

Use functions in 'simulation.py' to generate various datasets:

```python
basic_data = generateSample_simBasic(n=1000)
smc_data = generateSample_simSMC(n=187)
metabric_data = generateSample_simMETABRIC(n=1096)

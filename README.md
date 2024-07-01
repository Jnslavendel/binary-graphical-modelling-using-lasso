# Binary Graphical Modelling Using LASSO

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
- `usage.py`: Integrates various functionalities to directly replicate paper results.

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
    git clone https://github.com/Jnslavendel/binary-graphical-modelling-using-lasso.git
    cd graphical-modelling-lasso
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 

All the functionalities below can be run by placing the appropriate function in 'main.py'.

### Generating Simulation Data

To generate the various simulated datasets, use the following functions from `simulation.py`:

```python
basic_data = simulation.generateSample_simBasic(n=1000)
smc_data = simulation.generateSample_simSMC(n=187)
metabric_data = simulation.generateSample_simMETABRIC(n=1096)
```

### Data Processing

To preprocess the empirical datasets, use the functions in 'data_processing.py':

```python
smc_data = data_processing.load_data("data-external/BRCA-SMC-2018-clinical-data.tsv")
smc_data_processed = data_processing.preprocess_data_SMC(smc_data)

metabric_data = data_processing.load_data("data-external/BRCA-METABRIC-2012-2016-clinical-data.tsv")
metabric_data_processed = data_processing.preprocess_data_METABRIC(metabric_data)
```

### Graphical Modelling
The neighbourhoods for the graphical model can be constructed using LASSO-CV, LASSO-AIC, LASSO-BIC, or BOLASSO-CV. These functionalities are in 'graphical_modelling.py' and can be used as follows:

```python
neighbourhoods = graphical_modelling.findNeighbourhoodsLASSO_CV(basic_data, max_lambda=200)
```

Here, the neighbourhoods of all categorical dummy variables can be reverted to their original categorical variables using the following function:

```python
generalised_neighbourhoods = graphical_modelling.generaliseNeigbourhoodsToCategories(neighbourhoods)
```

### Visualising Graphs
To visualise the constructed graphical models with the found neighbourhoods, use the functions in graph_drawing.py, which rely on the AND-rule or OR-rule:

```python
G_AND = graph_drawing.constructGraph_AND(neighbourhoods)
G_OR = graph_drawing.constructGraph_OR(neighbourhoods)
```

### Evaluation Models
Evaluate the performance of (simulation) graphical models using functions in evaluation.py, as shown below:

```python
known_structure = evaluation.create_known_structure_matrix_simBasic()
identified_structure = evaluation.create_identified_structure_matrix(G_AND, ["X1", "X2", "X3", "X4", "X5"])

SDH, YOUDEN = evaluate_SDH_YOUDEN(known_structure, identified_structure)
```

## Reproducing Paper Results

'usage.py' contains several functions to directly reproduce the results in the paper.

First, to run and evaluate a simulation (simulation_method) with a number of replications (nsim), sample size (nsample), search area (max_lambda), and graphical model (model), use the following function:
```python
usage.run_simulations(nsim=125, nsample=500, max_lambda=200, model='lasso-aic', simulation_method='basic', output_file='output/basicsim-results-lasso-aic-500')
```

To generate a boxplot with the output file above, use one of the create_boxplots_*() functions. For example: 

```python
usage.create_boxplots_basic_simulations(filepath=''output/basicsim-results-lasso-aic-500')
```

To directly create graphical models for the empirical METABRIC and SMC datasets, use the following functions:
```python
usage.run_empirical_metabric()
usage.run_emperical_smc()
```


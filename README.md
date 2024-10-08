[![DOI](https://zenodo.org/badge/745174549.svg)](https://doi.org/10.5281/zenodo.13870503)


# bonney_et-al_2024_jgrml
**Emulation of Monthly Water Allocations Using LSTM Models: A Case Study of the Colorado River Basin in Texas**

Kirk Bonney<sup>1\*</sup>, Thushara Gunda<sup>1</sup>, Stephen Ferencz<sup>2</sup>, and Nicole D. Jackson<sup>1</sup>

<sup>1 </sup> Sandia National Laboratories, Albuquerque, NM, USA
<sup>2 </sup> Pacific Northwest National Laboratory, Richland, WA, USA

\* corresponding author: klbonne@sandia.gov

## Abstract
As concerns about water scarcity increase due to a changing climate and extreme weather, there is a need for computational test beds to help explore water management challenges and elucidate interdependences within coupled human-natural systems. Prior approaches for water management analyses have ranged from process-based simulations to data-driven modeling. However, development of data-driven models for water availability can be challenging due to poor data availability and software limitations. To overcome these challenges, this work aims to develop a machine learning, long short-term model (LSTM)-based surrogate of a parsimonious process-based water management model, the Water Rights Analysis Package (WRAP). Synthetic streamflows were generated to create an ensemble of scenarios across a range of hydrologic conditions; these data were inputted into WRAP to obtain associated water allocation data for the LSTM training, testing, and validation. We applied this computational framework to a case study of the Colorado River Basin in Texas. The trained LSTM emulates the water allocation process with low overall error and demonstrates minimal performance impact when out-of-distribution drought scenarios are used. Exploratory analysis of error patterns shows that the LSTM effectively captures the spatio-temporal water allocation patterns well and has no significant bias for underlying system attributes. These findings suggest that LSTMs have the potential to serve as surrogates for water management models. The computational framework developed could be used to explore water management scenarios for other basins to understand potential impacts from increasingly extreme hydrologic conditions. 

## Journal reference
TBD

## Code reference
Bonney, K., Gunda, T., Ferencz, S., & Jackson, N. (2024). Supporting code for Bonney et al. - JGR: Machine Learning and Computation (0.0.1)[Code]. Zenodo. https://doi.org/10.5281/zenodo.13870503

## Data reference
| Dataset                                                                          | Link                                                                                          | DOI              |
|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------|
| Synthetic Streamflow Datasets to Support Emulation of Water Allocations via LSTM | https://data.msdlive.org/records/cfj8d-xsb13                                                  | 10.57931/2441443 |
| Water Availability Model for the Colorado River Basin                            | https://www.tceq.texas.gov/permitting/water_rights/wr_technical-resources/wam.html            | n/a              |
| Water Rights for the Colorado River Basin                                        | https://tceq.maps.arcgis.com/apps/webappviewer/index.html?id=44adc80d90b749cb85cf39e04027dbdc | n/a              |

## Reproduce this work
Clone this repository (`git clone https://github.com/IMMM-SFA/bonney_et-al_2024_jgrml.git`) and install the `toolkit` package into a Python 3.11 environment (`pip install -e .`). Copy the repo_data/ folder from the accompanying [MSD-Live archive](https://data.msdlive.org/records/cfj8d-xsb13) as a top level directory in the repository. Once the environment is established, this work can be reproduced by running scripts from the workflow/ directory. There are three subdirectories which correspond to different stages of the experiment:

| Directory name        | Description                                                                                                                                                                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| I_Dataset_Generation/ | This directory contains scripts for generating synthetic streamflows and corresponding water shortage outputs from WRAP. These realizations are processed for machine learning and bundled into datasets for training/validation and testing. |
| II_Machine_Learning/  | This directory contains scripts for optimizing and training an LSTM model on the training/validation dataset and predicting outputs on the testing datasets.                                                                                  |
| III_Results/          | This directory contains scripts for evaluating the performance of the machine learning model on the testing datasets and generating figures used in the publication.                                                                          |

Generally, all scripts have two header sections that the user will customize based on which datasets they wish to produce/use. The first is indicated by "## Settings ##" and allows the user to customize variables such as random seeds and parameters of the experiment. A markdown file called `script_params.md` is located in the workflow folder to indicate which settings to use to reproduce the experiment and figures used in the paper. The second section is indicated by "## Path configuration ##" and includes paths to various input and output folders used in the script. The intent is that the user should not need to customize these paths as long as they copy the repo_data/ directory into the repository. Note that a large portion of the code is in functions and classes defined in the `toolkit` package included in the repsitory and users looking for details should explore this package in addition to the scripts.

### Reproduce datasets
The workflow in `I_Dataset_Generation/` uses hidden markov models to generate synthetic streamflows which are used as inputs to the water rights analysis package (WRAP) to obtain corresponding water shortage outputs for each scenario. Then the input and output datasets are processed and combined into a single dataset for use in the machine learning workflow.

This workflow is computationally expensive, as it involves running the WRAP simulation executable 1500 times to create all datasets. There is code for multiprocessing, but a single run of WRAP utilizes a large amount of memory so RAM will be a limiting factor. With 32G of RAM 4 processes were able to be concurrently run and took 2-3 days to complete on an Intel i7 processor. With lower RAM and fewer concurrent processes, this could take much longer. Additionally, the code for executing WRAP is designed to run on a Linux operating system using the Windows emulation software [https://www.winehq.org/](Wine). Running the script on Windows or Mac would require edits to the code. If you need help with this, the authors are happy to work with you via GitHub issues to get the code running on your OS. For those who primarily wish to reproduce the machine learning results and figures, data outputs from this section are provided in the `repo_data/ml-data/` folder so that running these scripts may be skipped. To use these data outputs instead of regenerating them, move them to the `outputs/` folder.

| Script name                         | Description                                                                                                                                                                                                                                                                                                            |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1_generate_synthetic_streamflows.py | This script generates synthetic streamflow datasets using historically observed flows to train an HMM model that can produce synthetic flows. The script must be run a separate time for each dataset (1 for training and 5 for testing), the parameters for these datasets can be found in the script_params.md file. |
| 2_simulate_water_shortages.py       | This script executes the WRAP simulation using synthetic streamflows to produce corresponding water shortage outputs. The script must be run a separate time for each dataset (1 for training and 5 for testing), the parameters for these datasets can be found in the script_params.md file.                         |
| 3_process_datasets_for_ml.py        | This script processes the synthetic streamflows and shortages in preparation for machine learning and combines them into a single dataset. The script is set up to run a single time once all 6 datasets have been generated using the previous two steps. No random seed is used in this script.                      |


### Reproduce modelling
The workflow in `II_Machine_Learning/` uses synthetic datasets to develop an LSTM model to predict water shortages across rights using the streamflow conditions across gage sites. The workflow uses the datasets either produced by the previous workflow or those provided in the `repo_data/ml-data/` directory from MSDLive. If you are skipping dataset generation you will need to copy `repo_data/ml-data/synthetic-trainvalid/` and `repo_data/ml-data/synthetic-test/` into the `outputs/` directory.

The `1_optimize_model.py` script is computationally expensive, especially if a GPU is not available, and can be skipped as the final parameters used to train the model used in the paper are saved in `repo_data/ml-configs/optimized_params.yaml`.

| Script name               | Description                                                                                                                                                        |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1_optimize_model.py       | This script searches for optimal parameters for the model using the training/validation dataset and saves the m to a json file.                                                                   |
| 2_train_model.py          | This script trains the model using the provided parameters with an 80/20 split of the training/validation data. By default, the script will use optimized_params.yaml to set up the model, which will reproduce the LSTM trained for the paper.                                                                                                         |
| 3_generate_predictions.py | This script loads a model at the selected epoch to generate predictions on the test datasets which are then used for evaluation of the model in the next workflow. |

### Reproduce figures
The scripts in the `III_Results/` workflow reproduce the figures of the associated paper, given that the previous workflows were followed. The first script roughly corresponds to figures/tables in the methods section, the second to figures in the supplementary materials, the third to elements of the flowchart found in the methods section, and the fourth to the results section. Figures and tables are saved to the `outputs/figures/` directory. Training the model from scratch can be skipped by moving the `repo_data/ml-models/run_20240916-123332` folder to `outputs/runs/run_20240916-123332`.

| Script name                    | Description                                                                                                                                                          |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1_plot_datasets.py             | This script produces various figures representing the streamflow and shortage datasets.                                                                              |
| 2_plot_metadata.py             | This script produces various figures and tables visualizing and summarizing the metadata associated to the water rights.                                             |
| 3_plot_flowchart_components.py | This script produces various figure elements that are used to build the flowchart diagram in the paper.                                                              |
| 4_plot_results.py              | This script uses the model predictions from the previous workflow to compute error metrics and produce tables and figures for the results section of the manuscript. |

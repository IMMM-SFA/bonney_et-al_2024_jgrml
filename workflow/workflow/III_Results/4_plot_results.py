from os.path import join
import os
import pandas as pd
from toolkit.utils.io import hdf5_to_dict
from toolkit.graphics.model import generate_results, process_errors, process_streamflow
from toolkit import outputs_data_path


## Settings ##
# select model run and checkpoint
run_name = "run_20240916-123332"
checkpoint_id = "249"

## Path Configuration ##

synthetic_test_data_path = join(outputs_data_path,
    "synthetic-test")

figure_output_path = join(outputs_data_path, "figures", "results", run_name)
if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)

## Main Script ##

# Generate results for each test dataset and store high level summaries in dictionaries for 
# later use
metrics = {}
volumetric_metrics = {}
drought_levels = [0.0,0.1,0.2,0.3,0.4,0.5]
for drought in drought_levels:
    drought_label = str(int(drought*100))+"%"
    
    # create dataset folders
    dataset_outputs_folder = join(figure_output_path, drought_label)
    os.makedirs(dataset_outputs_folder, exist_ok=True)
    
    # load dataset and predictions
    dataset_path = join(
        synthetic_test_data_path,
        f"synthetic_test_dataset_drought_{str(drought)}.h5"
    )
    
    pred_path = join(
    synthetic_test_data_path,
    f"synthetic_test_dataset_drought_{str(drought)}_{run_name}_predictions.h5"
    )
    
    data_dict = hdf5_to_dict(dataset_path)
    right_labels = list(data_dict["shortage_columns"])
    right_sectors = list(data_dict["sector"])
    right_seniority = list(data_dict["seniority"])
    right_allotment = list(data_dict["allotment"])
    time_index = pd.to_datetime(list(data_dict["shortage_index"]))
    
    pred_dict = hdf5_to_dict(pred_path)
    inputs = data_dict["streamflow_data"]
    true_outputs = data_dict["shortage_data"]
    predictions = pred_dict["shortage_predictions"]
    
    results_tuple = (inputs, true_outputs, predictions)

    # generate errors
    errors_dict = process_errors(        
        results_tuple,
        right_labels,
        right_sectors,
        right_seniority,
        right_allotment,
        time_index)
    
    # save overall df for summary tables
    metrics[drought_label] = errors_dict["overall_df"]
    
    # process streamflows
    streamflow_df = process_streamflow(results_tuple, aggregation="years")
    
    # generate tables and plots
    generate_results(
        errors_dict,
        streamflow_df,
        dataset_outputs_folder)

## Generate tables to summarize error across drought parameter

# helper function for table formatting
def round_to_3(x):
    return f"{x:.3f}"

# drought scenarios shortage ratio table
data = {}
for dataset_name, dataset_metrics in metrics.items():
    data[dataset_name] = metrics[dataset_name].loc["All rights ratio"]
drought_shortage_metrics_df = pd.DataFrame(data).T
formatters = {column: round_to_3 for column in drought_shortage_metrics_df.columns}
drought_shortage_metrics_df.to_latex(join(figure_output_path, "drought_shortage_error_summary_df.tex"), formatters=formatters)

# drought scenarios volumetric table
data = {}
for dataset_name, dataset_metrics in metrics.items():
    data[dataset_name] = metrics[dataset_name].loc["All rights volume"]
drought_volume_metrics_df = pd.DataFrame(data).T
formatters = {column: round_to_3 for column in drought_volume_metrics_df.columns}
drought_volume_metrics_df.to_latex(join(figure_output_path, "drought_volume_error_summary_df.tex"), formatters=formatters)

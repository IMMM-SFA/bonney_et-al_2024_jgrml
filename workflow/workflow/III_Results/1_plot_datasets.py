from os.path import join
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from toolkit.graphics.palette import DROUGHT_PALETTE, MEANPROPS
from toolkit.graphics.model import process_streamflow, process_shortage
from toolkit.utils.io import hdf5_to_dict
from toolkit import outputs_data_path


## Settings ##
aggregation = "years" # sets the axes along which aggregation occurs in later functions

# select model run and checkpoint
run_name = "run_20240916-123332"
checkpoint_id = "249"

## Path Configuration ##
synthetic_test_data_path = join(
    outputs_data_path,
    "synthetic-test")

figure_output_path = join(outputs_data_path, "figures", "datasets")
if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)

## Main Script ##

# Load test data, extract outflow streamflow, and extract ML results
test_datadict = dict()
streamflow_dfs = []
shortage_dfs = []
month_labels = [month[:3].upper() for month in calendar.month_name[1:]]
for drought in [0.0,0.1,0.2,0.3,0.4,0.5]:
    drought_label = f"{drought:.1f}"
    test_datadict[drought_label] = dict()
    
    # load dataset
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
    right_rights = list(data_dict["allotment"])
    time_index = pd.to_datetime(list(data_dict["shortage_index"]))
    inputs = data_dict["streamflow_data"]
    true_outputs = data_dict["shortage_data"]
    
    pred_dict = hdf5_to_dict(pred_path)
    predictions = pred_dict["shortage_predictions"]
    
    results_tuple = (inputs, true_outputs, predictions)
    test_datadict[drought_label]["results_tuple"] = results_tuple
        
    ## streamflow
    streamflow_df = process_streamflow(results_tuple, aggregation=aggregation)
    streamflow_df["drought"] = drought_label
    streamflow_dfs.append(streamflow_df)
    
    #shortage
    shortage_df = process_shortage(results_tuple, aggregation=aggregation)
    shortage_df["drought"] = drought_label
    shortage_dfs.append(shortage_df)


## Boxplots of streamflow levels at outflow node averaged across datasets 
ensemble_df = pd.concat(streamflow_dfs)
ensemble_df.rename(columns={"drought": "Drought Adjustment"}, inplace=True)

fig, ax = plt.subplots(figsize=(16,7))
sns.boxplot(
    ax=ax,
    x=ensemble_df["variable"],
    y=ensemble_df["value"],
    hue=ensemble_df["Drought Adjustment"],
    showfliers=False,
    meanprops=MEANPROPS,
    palette=DROUGHT_PALETTE,
    medianprops={'color': 'white', 'linewidth': 2},
)

ax.set_xlabel(None)
ax.set_ylabel("Average Streamflow (acre-feet)", fontsize=20)
ax.set_xticklabels(month_labels)
fig.tight_layout()

fig.savefig(join(figure_output_path, "drought_streamflow.png"))

## Boxplots of shortage levels averaged across datasets and rights
ensemble_df = pd.concat(shortage_dfs)
ensemble_df.rename(columns={"drought": "Drought\nAdjustment"}, inplace=True)
fig, ax = plt.subplots(figsize=(16,7))
sns.boxplot(
    ax=ax,
    x=ensemble_df["variable"],
    y=ensemble_df["value"],
    hue=ensemble_df["Drought\nAdjustment"],
    showfliers=False,
    meanprops=MEANPROPS,
    palette=DROUGHT_PALETTE,
    medianprops={'color': 'white', 'linewidth': 2},
    legend="brief"
)

ax.set_xlabel(None)
ax.set_ylabel("Average Shortage Ratio", fontsize=20)
ax.set_xticklabels(month_labels)
ax.set_ylim(0,1)

fig.tight_layout()

fig.savefig(join(figure_output_path, "drought_shortages.png"))

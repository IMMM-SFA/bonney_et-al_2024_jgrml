from os.path import join
import os
import pandas as pd
import numpy as np
import zipfile
import multiprocessing
from toolkit.utils.io import dict_to_hdf5, hdf5_to_dict
from toolkit.emulator.processing import process_diversion_csv
from toolkit.wrap.io import df_to_flo, flo_to_df
from toolkit.wrap.wrapdriver import WRAPDriver
from toolkit.wrap.io import out_to_csvs
from toolkit import repo_data_path, outputs_data_path
from toolkit.wrap.wraputils import clean_folders, split_into_subslits


## Settings ##
"""
This script runs the wet and dry (non-stationary) streamflow data through the water shortage simulation pipeline.
It uses the modified datasets created by 1_generate_nonstationary_streamflow.py as input.
The original dataset simulations already exist, so we only need to simulate the wet and dry scenarios.

To run this script efficiently, it is recommended to use the maximum number of processes that your system can handle.
One way to find this is to start with a large number of processes (8) and iteratively attempt to run the script while
dropping the number down by 1 each time. When using more than one process, the repo_data/wrap_execution_directories/execution_folder_0
directory needs to be duplicated for as many processes you intend to run, where the 0 increases to 1 and so on for each copy.
"""

# Pipeline control options
run_original_pipeline = True  # Set to False to skip original dataset simulation  
run_wet_pipeline = True   # Set to False to skip wet dataset simulation
run_dry_pipeline = True   # Set to False to skip dry dataset simulation

num_processes = 3
drought_level = 0.0  # Should match the drought level used in 1_generate_nonstationary_streamflow.py
# n_subset = 10  # Number of streamflows to process (matching script 1)

## Path configuration ##

# Use local outputs directory in the same folder as this script
local_outputs_path = join(os.path.dirname(__file__), "outputs")
if not os.path.exists(local_outputs_path):
    os.makedirs(local_outputs_path)

# Input: Original, wet, and dry subset datasets (from local outputs)
original_subset_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_original_subset.h5"
)

wet_subset_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_wet_subset.h5"
)

dry_subset_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_dry_subset.h5"
)

# Output: Shortage results for all three datasets (save to local outputs)
original_shortage_output_path = join(
    local_outputs_path,
    f"shortages_drought_{str(drought_level)}_original_subset.h5"
)

wet_shortage_output_path = join(
    local_outputs_path,
    f"shortages_drought_{str(drought_level)}_wet_subset.h5"
)

dry_shortage_output_path = join(
    local_outputs_path,
    f"shortages_drought_{str(drought_level)}_dry_subset.h5"
)

# FLO file paths (use local outputs)
wrap_flo_path = join(repo_data_path, "colorado-full", "C3.FLO")
original_flo_output_path = join(local_outputs_path, "original_flos")
wet_flo_output_path = join(local_outputs_path, "wet_flos")
dry_flo_output_path = join(local_outputs_path, "dry_flos")

# WRAP file paths
wrap_execution_path = join(repo_data_path, "wrap_execution_directories")
wrap_sim_path = join(wrap_execution_path, "SIM.exe")
original_out_zip_path = join(local_outputs_path, "original_out_zips")
wet_out_zip_path = join(local_outputs_path, "wet_out_zips")
dry_out_zip_path = join(local_outputs_path, "dry_out_zips")

# water shortage raw csv output (use local outputs)
original_shortage_csvs_path = join(local_outputs_path, "original_shortage_csvs")
wet_shortage_csvs_path = join(local_outputs_path, "wet_shortage_csvs")
dry_shortage_csvs_path = join(local_outputs_path, "dry_shortage_csvs")

# ensure necessary directories exist
all_directories = [
    original_flo_output_path, wet_flo_output_path, dry_flo_output_path,
    original_shortage_csvs_path, wet_shortage_csvs_path, dry_shortage_csvs_path,
    original_out_zip_path, wet_out_zip_path, dry_out_zip_path
]
for directory_path in all_directories:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# clean out folders from previous runs
clean_folders(wrap_execution_path, original_shortage_csvs_path, original_flo_output_path)
clean_folders(wrap_execution_path, wet_shortage_csvs_path, wet_flo_output_path)
clean_folders(wrap_execution_path, dry_shortage_csvs_path, dry_flo_output_path)

## Main Script ##

print("="*80)
print("PROCESSING ORIGINAL, WET AND DRY NON-STATIONARY SUBSET DATASETS")
print("="*80)

# Check if the required datasets exist based on pipeline options
if run_original_pipeline and not os.path.exists(original_subset_path):
    raise FileNotFoundError(f"Original subset dataset not found: {original_subset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

if run_wet_pipeline and not os.path.exists(wet_subset_path):
    raise FileNotFoundError(f"Wet subset dataset not found: {wet_subset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

if run_dry_pipeline and not os.path.exists(dry_subset_path):
    raise FileNotFoundError(f"Dry subset dataset not found: {dry_subset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

def create_flo_files(dataset_path, flo_output_path, dataset_name):
    """Create FLO files for a given dataset"""
    print(f"\nCreating FLO files for {dataset_name} dataset...")
    print(f"Input: {dataset_path}")
    
    # Load dataset
    data_dict = hdf5_to_dict(dataset_path)
    streamflow_data = data_dict["streamflow_data"]
    streamflow_index = data_dict["streamflow_index"] 
    streamflow_columns = data_dict["streamflow_columns"]
    
    print(f"Dataset shape: {streamflow_data.shape}")
    
    # Load default FLO file
    default_flo = flo_to_df(wrap_flo_path)
    
    # Generate FLO files
    num_datapoints = streamflow_data.shape[0]
    print(f"Creating {num_datapoints} FLO files...")
    
    for i in range(num_datapoints):
        # Create synthetic flow DataFrame
        synth_flow = pd.DataFrame(
            streamflow_data[i,:,:], 
            index=streamflow_index,
            columns=streamflow_columns,
        )
        synth_flow.index = pd.to_datetime(synth_flow.index)
        default_flo.index = synth_flow.index
        
        # Add in historical flow at these two gage sites since they are outside of the CRB
        synth_flow["L10000"] = default_flo["INL10000"].astype(float)
        synth_flow["L20000"] = default_flo["INL20000"].astype(float)
        
        out_name = join(flo_output_path, f"{dataset_name}_flow_{i:02d}.FLO")
        df_to_flo(synth_flow, out_name)
    
    print(f"{dataset_name} FLO files created successfully.")
    return num_datapoints

# Create FLO files for enabled datasets
original_num_files = 0
wet_num_files = 0
dry_num_files = 0

if run_original_pipeline:
    original_num_files = create_flo_files(original_subset_path, original_flo_output_path, "original")

if run_wet_pipeline:
    wet_num_files = create_flo_files(wet_subset_path, wet_flo_output_path, "wet")
    
if run_dry_pipeline:
    dry_num_files = create_flo_files(dry_subset_path, dry_flo_output_path, "dry")

# Define a pipeline function to be utilized by multiprocessing
def wrap_pipeline(flo_files, wrap_execution_folder, flo_output_path, shortage_csvs_path, out_zip_path):
    """For each FLO file: copy FLO file to execution folder, run wrap, 
    process the OUT file, compresses the original OUT file, and delete the 
    OUT, MSS, and FLO file for the run.

    Parameters
    ----------
    flo_files : list[str]
        list of .FLO file paths
    wrap_execution_folder : str
        folder that pipeline will run wrap inside of
        outputs are left in this folder as well.
        Needs to contain the 5 configuration files for
        running WRAP.
    flo_output_path : str
        path to FLO files directory
    shortage_csvs_path : str
        path to save shortage CSV files
    out_zip_path : str
        path to save compressed OUT files
    """
    driver = WRAPDriver(wrap_sim_path)
    count = 0
    for flo_file in flo_files:
        # copy flo file to execution folder
        flo_name = flo_file.split(".")[0]
        flo_file = os.path.join(flo_output_path, flo_file)
        
        # execute wrap
        driver.execute(flo_file=flo_file,
                    execution_folder=wrap_execution_folder)
        
        # process .OUT file
        out_file = os.path.join(wrap_execution_folder, f"{flo_name}.OUT")
        mss_file = os.path.join(wrap_execution_folder, f"{flo_name}.MSS")
        out_to_csvs(out_file, wrap_execution_folder, csvs_to_write=["diversions"])
        
        # process diversion file
        diversions_path = os.path.join(wrap_execution_folder, f"{flo_name}_diversions.csv")
        diversions_df = pd.read_csv(diversions_path)
        processed_shortages = process_diversion_csv(diversions_df)
        processed_shortages.to_csv(os.path.join(shortage_csvs_path, f"{flo_name}_shortage.csv"))
        
        # compress out file
        zip_file = os.path.join(out_zip_path, f"{flo_name}.OUT.zip")
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as myzip:
            myzip.write(out_file)

        # delete files
        os.remove(out_file)
        os.remove(mss_file)
        os.remove(diversions_path)
        count += 1
        print(count, flo_file, f"process: {wrap_execution_folder[-1]}")


def run_wrap_pipeline(dataset_name, flo_output_path, shortage_csvs_path, out_zip_path, shortage_output_path):
    """Run WRAP pipeline for a specific dataset"""
    print(f"\n{'='*60}")
    print(f"RUNNING WRAP PIPELINE FOR {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # obtain list of flo files and split into sublists based on the number of processes
    flo_files = os.listdir(flo_output_path)
    flo_files.sort()
    sub_lists = split_into_subslits(flo_files, num_processes)
    
    print(f"Processing {len(flo_files)} FLO files with {num_processes} processes...")
    
    # run wrap pipeline across multiple processes
    processes = []
    for process_id, flo_file_list in enumerate(sub_lists):
        process_wrap_execution_folder = join(wrap_execution_path, f"execution_folder_{process_id}")
        process = multiprocessing.Process(
            target=wrap_pipeline, 
            args=(flo_file_list, process_wrap_execution_folder, flo_output_path, shortage_csvs_path, out_zip_path)
        )
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    print(f"WRAP simulations completed for {dataset_name} dataset.")
    
    ## Aggregate shortage csvs into hdf5 ##
    print(f"Aggregating {dataset_name} shortage results...")
    
    # get list of shortage files (ensure they are in the same order as streamflows)
    shortage_file_list = list(os.listdir(shortage_csvs_path))
    shortage_file_list.sort(key=lambda string: int(string.split("_")[2]))  # adjusted for "dataset_flow_XX"
    
    if not shortage_file_list:
        raise RuntimeError(f"No shortage files found in {shortage_csvs_path}")
    
    # first load example shortage df to get data shape
    example_shortage_df = pd.read_csv(join(shortage_csvs_path, shortage_file_list[0]), index_col=0)
    
    # initialize full shortage array
    shortages = np.zeros((len(shortage_file_list), example_shortage_df.shape[0], example_shortage_df.shape[1]))
    
    # iteratively load shortage values and insert into shortage array
    for i in range(len(shortage_file_list)):
        shortage_file = shortage_file_list[i]
        if shortage_file.endswith("_shortage.csv"):
            shortage_df = pd.read_csv(join(shortage_csvs_path, shortage_file), index_col=0)
            shortages[i,:,:] = shortage_df.values
    
    # create data dictionary containing data values, columns, and index
    data_dictionary = {}
    data_dictionary["shortage_data"] = shortages
    data_dictionary["shortage_index"] = list(shortage_df.index.astype(str))
    data_dictionary["shortage_columns"] = list(shortage_df.columns.astype(str))
    
    # write hdf5 file
    print(f"Saving {dataset_name} shortage results to: {shortage_output_path}")
    dict_to_hdf5(shortage_output_path, data_dictionary)
    
    print(f"{dataset_name} shortage data shape: {shortages.shape}")
    return shortages.shape

## Run pipelines for enabled datasets ##
print(f"\nRunning WRAP simulation pipeline with {num_processes} processes...")
print(f"Original pipeline: {'ENABLED' if run_original_pipeline else 'DISABLED'}")
print(f"Wet pipeline: {'ENABLED' if run_wet_pipeline else 'DISABLED'}")
print(f"Dry pipeline: {'ENABLED' if run_dry_pipeline else 'DISABLED'}")

original_shape = None
wet_shape = None
dry_shape = None

# Process original dataset
if run_original_pipeline:
    original_shape = run_wrap_pipeline(
        "original", 
        original_flo_output_path, 
        original_shortage_csvs_path, 
        original_out_zip_path, 
        original_shortage_output_path
    )
else:
    print("Skipping original dataset pipeline (run_original_pipeline = False)")

# Process wet dataset
if run_wet_pipeline:
    wet_shape = run_wrap_pipeline(
        "wet", 
        wet_flo_output_path, 
        wet_shortage_csvs_path, 
        wet_out_zip_path, 
        wet_shortage_output_path
    )
else:
    print("Skipping wet dataset pipeline (run_wet_pipeline = False)")

# Process dry dataset
if run_dry_pipeline:
    dry_shape = run_wrap_pipeline(
        "dry", 
        dry_flo_output_path, 
        dry_shortage_csvs_path, 
        dry_out_zip_path, 
        dry_shortage_output_path
    )
else:
    print("Skipping dry dataset pipeline (run_dry_pipeline = False)")

print("\n" + "="*80)
print("WATER SHORTAGE SIMULATION COMPLETE!")
print("="*80)
print("Results saved:")

if run_original_pipeline:
    print(f"  Original dataset: {original_shortage_output_path}")
    print(f"  Original shape: {original_shape}")
else:
    print("  Original dataset: SKIPPED")

if run_wet_pipeline:
    print(f"  Wet dataset: {wet_shortage_output_path}")
    print(f"  Wet shape: {wet_shape}")
else:
    print("  Wet dataset: SKIPPED")

if run_dry_pipeline:
    print(f"  Dry dataset: {dry_shortage_output_path}")
    print(f"  Dry shape: {dry_shape}")
else:
    print("  Dry dataset: SKIPPED")

datasets_processed = []
if run_original_pipeline:
    datasets_processed.append("original")
if run_wet_pipeline:
    datasets_processed.append("wet")
if run_dry_pipeline:
    datasets_processed.append("dry")

if datasets_processed:
    print(f"\n{', '.join(datasets_processed).title()} dataset(s) are ready for ML prediction and comparison analysis!")
else:
    print("\nNo datasets were processed. Set run_original_pipeline, run_wet_pipeline and/or run_dry_pipeline to True.")
    
print("All shortage simulations use the correctly modified streamflow data.")

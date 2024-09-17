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
To run this script efficiently, it is recommended to use the maximum number of processes that your system can handle.
One way to find this is to start with a large number of processes (8) and iteratively attempt to run the script while
dropping the number down by 1 each time. When using more than one process, the repo_data/wrap_execution_directories/execution_folder_0
directory needs to be duplicated for as many processes you intend to run, where the 0 increases to 1 and so on for each copy.
"""
num_processes = 1 
drought = None # This is used to create the name of the file output by step 1.

## Path configuration ##
# Synthetic streamflow dataset
if drought is None:
    synthetic_streamflow_path = join(outputs_data_path, "synthetic-trainvalid", f"streamflow{'_drought_'+str(drought) if drought else ''}.h5")
    synthetic_shortage_output_path = join(outputs_data_path, "synthetic-trainvalid", f"shortages{'_drought_'+str(drought) if drought else ''}.h5")
else:
    synthetic_streamflow_path = join(outputs_data_path, "synthetic-test", f"streamflow{'_drought_'+str(drought) if drought else ''}.h5")
    synthetic_shortage_output_path = join(outputs_data_path, "synthetic-test", f"shortages{'_drought_'+str(drought) if drought else ''}.h5")

# FLO file paths
wrap_flo_path = join(repo_data_path, "colorado-full", "C3.FLO")
synthetic_flo_output_path = join(outputs_data_path, "synthetic_flos")

# WRAP file paths
wrap_execution_path = join(repo_data_path, "wrap_execution_directories")
wrap_sim_path = join(wrap_execution_path, "SIM.exe")
out_zip_path = join(outputs_data_path, "out_zips")

# water shortage raw csv output
shortage_csvs_path = join(outputs_data_path, "shortage_csvs")

# ensure necessary directories exist
for directory_path in [synthetic_flo_output_path, shortage_csvs_path, out_zip_path]:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# clean out folders from previous runs
clean_folders(wrap_execution_path, shortage_csvs_path, synthetic_flo_output_path)

## Main Script ##

## Create FLO files for WRAP execution 
# load synthetic streamflow data
streamflow_dict = hdf5_to_dict(synthetic_streamflow_path)

# Load default FLO file
default_flo = flo_to_df(wrap_flo_path)

# Generate FLO files
num_datapoints = streamflow_dict["streamflow_data"].shape[0]
for i in range(num_datapoints):
    name = f"hmmsynth_streamflow_{i:02d}.csv"
    data = streamflow_dict["streamflow_data"][i,:,:]
    synth_flow = pd.DataFrame(
        streamflow_dict["streamflow_data"][i,:,:], 
        index=streamflow_dict["streamflow_index"],
        columns=streamflow_dict["streamflow_columns"],
        )
    synth_flow.index = pd.to_datetime(synth_flow.index)
    default_flo.index = synth_flow.index
    
    # add in historical flow at these two gage sites since they are outside of the CRB
    synth_flow["L10000"] = default_flo["INL10000"].astype(float)
    synth_flow["L20000"] = default_flo["INL20000"].astype(float)
    
    out_name = join(synthetic_flo_output_path, f"synthflow_{i:02d}.FLO")
    df_to_flo(synth_flow, out_name)

# Define a pipeline function to be utilized by multiprocessing
def wrap_pipeline(flo_files, wrap_execution_folder):
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
    """
    driver = WRAPDriver(wrap_sim_path)
    count = 0
    for flo_file in flo_files:
        # copy flo file to execution folder
        flo_name = flo_file.split(".")[0]
        flo_file = os.path.join(synthetic_flo_output_path, flo_file)
        
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


## Run wrap pipeline with multiprocessing ##
# For this code to run correctly, `wrap_execution_folder` needs to point to a directory
# that contains multiple subdirectories named "execution_folder_i"
# where i ranges from 0 to the max number of processes you want to run.
# One such subdirectory is provided in the repo and can be copied as needed.
# Each of these directories should contain the following files 
# which are required to run wrap:
# 
# C3.DAT
# C3.DIS
# C3.EVA
# C3.FAD
# C3.HIS

# obtain list of flo files and split into sublists based on the number of processes
flo_files = os.listdir(synthetic_flo_output_path)
flo_files.sort()
sub_lists= split_into_subslits(flo_files, num_processes)

# run wrap pipeline across multiple processes
processes = []
for process_id, flo_file_list in enumerate(sub_lists):
    process_wrap_execution_folder = join(wrap_execution_path, f"execution_folder_{process_id}")
    process = multiprocessing.Process(target=wrap_pipeline, args=(flo_file_list, process_wrap_execution_folder))
    processes.append(process)
    process.start()

for process in processes:
    process.join()


## Aggregate shortage csvs into hd5f ##

# get list of shortage files (ensure they are in the same order as streamflows)
shortage_file_list = list(os.listdir(shortage_csvs_path))
shortage_file_list.sort(key=lambda string: int(string.split("_")[1]))

# first load example shortage df to get data shape
example_shortage_df = pd.read_csv(join(shortage_csvs_path, shortage_file_list[0]),index_col=0)

# initalize full shortage array
shortages = np.zeros((len(shortage_file_list), example_shortage_df.shape[0], example_shortage_df.shape[1]))

# iteratively load shortage values and insert into shortage array
for i in range(len(shortage_file_list)):
    shortage_file = shortage_file_list[i]
    if shortage_file.endswith("_shortage.csv"):
        shortage_df = pd.read_csv(join(shortage_csvs_path, shortage_file),index_col=0)
        shortages[i,:,:] = shortage_df.values

# create data dictionary containing data values, columns, and index
data_dictionary = {}
data_dictionary["shortage_data"] = shortages
data_dictionary["shortage_index"] = list(shortage_df.index.astype(str))
data_dictionary["shortage_columns"] = list(shortage_df.columns.astype(str))

#write hdf5 file
dict_to_hdf5(shortage_output_path, data_dictionary)

from os.path import join
import numpy as np
import random
from toolkit.wrap.streamflow import WRAPStreamFlow
from toolkit.utils.io import dict_to_hdf5
from toolkit import repo_data_path, outputs_data_path


## Settings ##
random_state = 42 # Settings for HMM random state. 
n_runs = 1000 # how many streamflow instances to generate
drought = None # drought parameter (a): None for no drought adjustment or a float between 0 and 1

n_years = 77 # 1940-2016 to match default wrap runs, do not modify if intending to use streamflows as input to wrap

## Path configuration ##
if drought is None:
    synthetic_streamflow_output_path = join(outputs_data_path, "synthetic-trainvalid", f"streamflow{'_drought_'+str(drought) if drought else ''}.h5")
else:
    synthetic_streamflow_output_path = join(outputs_data_path, "synthetic-test", f"streamflow{'_drought_'+str(drought) if drought else ''}.h5")

wrap_flo_path = join(repo_data_path, "colorado-full", "C3.FLO")

## Main Script ##

# The streamflow generation process is managed by the WRAPStreamFlow class. 
# Please refer to the class definition for implemenation details.
wrap_streamflow = WRAPStreamFlow(
    wrap_flo_path, 
    outflow_name="INK20000",
    ignore_columns=["INL10000", "INL20000"])
wrap_streamflow.load_streamflows(start_year=1979)

# Generate a reproducible list of random seeds for running the HMM, so that each run is distinct.
random.seed(random_state)
random_seeds = [random.randint(0,2**32-1) for _ in range(n_runs)]
streamflows=[]
for i in range(n_runs):
    sf = wrap_streamflow.generate_synthetic_streamflow(
        start_year="1940",
        num_years=n_years,
        drought=drought,
        random_seed=random_seeds[i])
    streamflows.append(sf.values)

# create data dictionary containing data values, columns, and index
streamflow_dict = {}
streamflow_dict["streamflow_data"] = np.array(streamflows)
streamflow_dict["streamflow_index"] = list(sf.index.astype(str))
streamflow_dict["streamflow_columns"] = list(sf.columns.astype(str))

# write to hdf5 file
dict_to_hdf5(synthetic_streamflow_output_path, streamflow_dict)

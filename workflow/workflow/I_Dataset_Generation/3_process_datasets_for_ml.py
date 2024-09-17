
from os.path import join
import numpy as np
import pandas as pd
from toolkit.utils.io import hdf5_to_dict, dict_to_hdf5
from toolkit import repo_data_path, outputs_data_path
from toolkit.data.data import filter_dataset


## Settings ## 

# filtering options
filter_sectorally = True
filter_seniority = True
filter_allotment = True
filter_geospatially = True
filter_variable_demand = True
filter_zero_allotment_rights = True

start_year = 1940
end_year = 2016

## Path Configuration ##

# synthetic training data
streamflow_path = join(outputs_data_path, "synthetic-trainvalid", "streamflow.h5")
shortage_path = join(outputs_data_path, "synthetic-trainvalid", "shortage.h5")

# output training dataset
output_training_dataset_path = join(outputs_data_path, "synthetic-trainvalid", "synthetic_trainvalid_dataset.h5")

# other datapaths
dat_path = join(repo_data_path, "colorado-full", "C3_processed_dat.csv")
historical_diversions_path = join(repo_data_path, "colorado-full", "C3_diversions.csv")
subordinate_rights_path = join(repo_data_path, "misc_data", "variable_demand_rights.csv")

## Main Script ##

#streamflow
streamflow_dict = hdf5_to_dict(streamflow_path)
            
#shortage
shortage_dict = hdf5_to_dict(shortage_path)

# extract data from dictionaries
streamflow_data = streamflow_dict["streamflow_data"]
streamflow_index = streamflow_dict["streamflow_index"]
streamflow_columns = streamflow_dict["streamflow_columns"]

shortage_data = shortage_dict["shortage_data"]
shortage_index = shortage_dict["shortage_index"]
shortage_columns = shortage_dict["shortage_columns"]

## Collect metadata attributes
dat_df = pd.read_csv(dat_path)
historical_diversions = pd.read_csv(historical_diversions_path)

# get right sector labels
right_sectors = dat_df.groupby("water_right_identifier").use.first()
assert (right_sectors.index == shortage_columns).all()

# get right seniority labels
right_seniority = dat_df.groupby("water_right_identifier").priority_number.first()
right_seniority = right_seniority.astype(str)

# convert to a datetime compatible format
year = right_seniority.str.slice(0,4)
month = right_seniority.str.slice(4,6)
day = right_seniority.str.slice(6,8)
datetimes = year+"-"+month+"-"+day
datetimes = datetimes.replace("1914-04-31", "1914-04-30") # fix a problematic date (4/31 does not exist)

right_seniority = pd.to_datetime(datetimes, errors="coerce")
assert (right_seniority.index == shortage_columns).all()

# get water right allotment sizes
historical_diversions["date"] = pd.to_datetime(historical_diversions[["year", "month"]].assign(DAY=1))

right_allotments = historical_diversions.pivot_table(
    index="date",
    columns="water_right_identifier",
    values="diversion_or_energy_target",
    dropna=False
)
right_allotments = right_allotments.iloc[0:12].sum() # sum over 12 month period
right_allotments = right_allotments.loc[shortage_columns]
assert (right_allotments.index == shortage_columns).all()
            
# build mask filters for columns 

# this list keeps track of all masks created across different filters
masks = []

if filter_sectorally:
    # clean up sector labels
    sector_categories = ["IND", "IRR", "MIN", "MUN", "POW", "REC"]
    def process_use(row):
        for sector in sector_categories:
            try:
                if sector in row:
                    return sector
            except TypeError:
                return np.nan

    right_sectors = right_sectors.apply(process_use)
    sectoral_mask = ~right_sectors.isna()
    masks.append(sectoral_mask)
    print("rights with no sector:", sum(~sectoral_mask))

if filter_seniority:
    seniority_mask = ~right_seniority.isna()
    masks.append(seniority_mask)
    print("rights with no seniority:", sum(~seniority_mask))
    
if filter_allotment:
    allotment_mask = ~right_allotments.isna()
    masks.append(allotment_mask)
    print("rights with no allotment:", sum(~allotment_mask))

if filter_geospatially:
    # load lat lon information
    latlong_gdf_path = join(repo_data_path, "geospatial", "wrap_right_latlon.csv")
    latlongs = pd.read_csv(latlong_gdf_path)
    latlongs.set_index("water_right_identifier", inplace=True)
    
    # identify rights that have lat/lon coordinates
    columns_with_latlong = latlongs.index
    geospatial_mask = pd.Series(shortage_columns).isin(columns_with_latlong).values
    masks.append(geospatial_mask)
    print("rights with no coordinates: ", sum(~geospatial_mask))


if filter_variable_demand:
    subordinate_rights = pd.read_csv(subordinate_rights_path, header=None)
    subordinate_filter = ~np.isin(shortage_columns, subordinate_rights)
    masks.append(subordinate_filter)
    print("rights with variable demand:", sum(~subordinate_filter))
    
if filter_zero_allotment_rights:
    zero_allotment_filter = ~(right_allotments==0)
    masks.append(zero_allotment_filter)
    print("rights with zero total allotment:", sum(~zero_allotment_filter))
    

# Convert NaN to 0
shortage_data[np.isnan(shortage_data)] = 0.0

## Apply the masks
full_mask = np.logical_and.reduce(masks)

shortage_data = shortage_data[:,:,full_mask]
shortage_columns = shortage_columns[full_mask]
right_sectors = right_sectors[full_mask]
right_seniority = right_seniority[full_mask]
right_allotments = right_allotments[full_mask]

# Assert that data structures shapes line up as a final check
assert streamflow_data.shape[1] == streamflow_index.shape[0]
assert streamflow_data.shape[2] == streamflow_columns.shape[0]

assert shortage_data.shape[1] == shortage_index.shape[0]
assert shortage_data.shape[2] == shortage_columns.shape[0]
assert shortage_data.shape[2] == right_sectors.shape[0]
assert shortage_data.shape[2] == right_seniority.shape[0]
assert shortage_data.shape[2] == right_allotments.shape[0]

assert shortage_data.shape[0] == streamflow_data.shape[0]
assert shortage_data.shape[1] == streamflow_data.shape[1]

assert (streamflow_index == shortage_index).all()

# Change metadata to hdf5 acceptable dtype
right_sectors = right_sectors.astype(str)
right_seniority = right_seniority.astype(str)

# Write data to hdf5
data_dict = dict() 
data_dict["streamflow_data"] = streamflow_data
data_dict["streamflow_index"] = streamflow_index
data_dict["streamflow_columns"] = streamflow_columns

data_dict["shortage_data"] = shortage_data
data_dict["shortage_index"] = shortage_index
data_dict["shortage_columns"] = shortage_columns
data_dict["sector"] = right_sectors.values
data_dict["seniority"] = right_seniority.values
data_dict["allotment"] = right_allotments.values

print("Final streamflow data shape:", streamflow_data.shape)
print("Final shortage data shape:", shortage_data.shape)

print("train/valid", "\n", "median streamflow:", 
      np.mean(streamflow_dict["streamflow_data"]), "\n", "mean streamflow:", 
      np
      .median(streamflow_dict["streamflow_data"]))

dict_to_hdf5(output_training_dataset_path, data_dict)

## process test datasets
# The testing datasets are filtered by applying the mask produced for the training dataset
# rather than redoing the entire filtering process.

# synthetic testing
for drought in [0.0,0.1,0.2,0.3,0.4,0.5]:
    streamflow_path = join(outputs_data_path, "synthetic-test", f"streamflow_drought_{str(drought)}.h5")
    shortage_path = join(outputs_data_path, "synthetic-test", f"shortages_drought_{str(drought)}.h5")

    streamflow_dict = hdf5_to_dict(streamflow_path)
    shortage_dict = hdf5_to_dict(shortage_path)

    test_data_dict = filter_dataset(streamflow_dict, shortage_dict, data_dict)

    output_test_dataset_path = join(outputs_data_path, "synthetic-test", f"synthetic_test_dataset_drought_{drought}.h5")
    
    print(f"drought_{drought}", "\n", "median streamflow:", 
      np.mean(streamflow_dict["streamflow_data"]), "\n", "mean streamflow:", 
      np.median(streamflow_dict["streamflow_data"]))
    
    dict_to_hdf5(output_test_dataset_path, test_data_dict)


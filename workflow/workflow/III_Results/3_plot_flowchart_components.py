from os.path import join
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from toolkit.utils.io import hdf5_to_dict
from toolkit import repo_data_path, outputs_data_path

"""
Produce examples of historical and synthetic annual outlet 
gage streamflow and gage site locations for use in flowchart.
"""

## Settings ##

## Path Configuration ##

historical_dataset_path = join(
    repo_data_path,
    "ml-data",
    "historical",
    "historic_dataset.h5"
    )

synthetic_dataset_path = join(
    outputs_data_path,
    "synthetic-test",
    f"synthetic_test_dataset_drought_{str(0.0)}.h5"
    )

crb_gdf_path = join(repo_data_path, "geospatial", "CRB")
flowline_gdf_path = join(repo_data_path, "geospatial", "Flowline")
gage_gdf_path = join(repo_data_path, "geospatial", "Gages")

figure_output_path = join(outputs_data_path, "figures", "flowchart")
if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)

## Main Script ##

historical_dataset = hdf5_to_dict(historical_dataset_path)
synthetic_dataset = hdf5_to_dict(synthetic_dataset_path)

# load in crb
crb = gpd.read_file(crb_gdf_path)

# load in flowlines
flowline = gpd.read_file(flowline_gdf_path)
flowline.to_crs(crb.crs, inplace=True)

# load in gages
gages = gpd.read_file(gage_gdf_path)
gages.to_crs(crb.crs, inplace=True)

# historical streamflow example
historical_streamflow = historical_dataset["streamflow_data"]
outlet_streamflow = historical_streamflow[:,:,42]
yearly_outlet_streamflow = outlet_streamflow.reshape(-1,12).sum(axis=1)

fig, ax = plt.subplots(figsize=(8,7))

ax.plot(yearly_outlet_streamflow, linewidth=10)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(join(figure_output_path, "historical_annual_outlet.png"))

# synthetic streamflow examples
synthetic_streamflow = synthetic_dataset["streamflow_data"]
for i in range(3):
    outlet_streamflow = synthetic_streamflow[i,:,42]
    yearly_outlet_streamflow = outlet_streamflow.reshape(-1,12).sum(axis=1)
    fig, ax = plt.subplots(figsize=(8,7))
    ax.plot(yearly_outlet_streamflow, linewidth=10, color="lightblue")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(join(figure_output_path, f"synthetic_annual_outlet_{i}.png"))
    
    
# basin with gage sites
fig, ax = plt.subplots()
gages.plot(ax=ax, color="red", markersize=120)
crb.plot(ax=ax, zorder=-1, edgecolor="black", linewidth=2, facecolor="none")
flowline.plot(ax=ax, zorder=-1, edgecolor="blue", linewidth=0.5)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(join(figure_output_path, "gage_locations.png"))

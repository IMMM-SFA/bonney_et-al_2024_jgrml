from os.path import join
import os
import calendar
import geopandas as gpd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from toolkit.utils.io import hdf5_to_dict
from toolkit.emulator.dataset import WrapDataset
from toolkit.utils.io import load_right_latlongs
from toolkit.graphics.palette import SECTOR_COLORS
from toolkit import repo_data_path, outputs_data_path


## Settings ##

## Path Configuration ##
dat_path = join(repo_data_path, "colorado-full", "C3_processed_dat.csv")

# path to one of the datasets to extract metadata from (can be training, test, or historical)
dataset_path = join(
    outputs_data_path,
    "synthetic-trainvalid",
    "synthetic_trainvalid_dataset.h5"
    )

figure_output_path = join(outputs_data_path, "figures", "metadata")
if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)
    
## Main Script ##

# load data
dataset = WrapDataset(hdf5_to_dict(dataset_path))

# extract data
right_labels = dataset.data_dict["shortage_columns"]
right_sectors = dataset.data_dict["sector"]
right_seniority = dataset.data_dict["seniority"]
right_size = dataset.data_dict["allotment"]

# create dataframe for water right metadata
df = gpd.GeoDataFrame(index=right_labels)
df["sector"] = right_sectors
df["seniority"] = pd.to_datetime(right_seniority)
df["allotment"] = right_size

# load in latlongs as point geometries for water rights
latlongs = load_right_latlongs()
latlongs = latlongs.loc[right_labels]
df["geometry"] = latlongs.geometry
df.set_geometry("geometry")

# load in crb
crb_gdf_path = join(repo_data_path, "geospatial", "CRB")
crb = gpd.read_file(crb_gdf_path)
crb.to_crs(latlongs.crs, inplace=True)

# load in flowlines
flowline_gdf_path = join(repo_data_path, "geospatial", "Flowline")
flowline = gpd.read_file(flowline_gdf_path)
flowline.to_crs(latlongs.crs, inplace=True)

# load allotments using historical diversion output
dat_df = pd.read_csv(dat_path)

historical_diversions = pd.read_csv(join(repo_data_path, "colorado-full", "C3_diversions.csv"))
historical_diversions["date"] = pd.to_datetime(historical_diversions[["year", "month"]].assign(DAY=1))

right_allotments = historical_diversions.pivot_table(
    index="date",
    columns="water_right_identifier",
    values="diversion_or_energy_target",
    dropna=False
)
monthly_right_allotment = right_allotments.iloc[0:12]

# compute log of right size
df["log_allotment"] = np.log10(df.allotment)

# load use patterns
use_pattern_path = join(repo_data_path, "misc_data", "wrap_monthly_demand_curves.csv")
use_patterns_df = pd.read_csv(use_pattern_path)

## Plot metadata

# distribution of water right sectors after filtering
print(pd.Series(right_sectors).value_counts())

# plot examples of monthly use patterns
fig, ax = plt.subplots()
for i, pattern_row in use_patterns_df.iterrows():
    sector = pattern_row.Sector.upper()
    if "IRR" in sector:
        sector = "IRR"
    data = pattern_row[1:13]
    if sector == "MIN":
        ax.plot(data, color=SECTOR_COLORS[sector], linewidth=5, label=sector, linestyle="dashed", zorder=5)
    else:
        ax.plot(data, color=SECTOR_COLORS[sector], linewidth=5, label=sector)
    
ax.set_xticks(range(0,12))
ax.set_xticklabels([mon[:3] for mon in calendar.month_name[1:]], rotation=45, ha='right')
ax.set_ylabel("Monthly Proportion of Total Allotment")  

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# adjust legend to remove duplicates
unique_labels = set(labels)
legend_dict = dict(zip(labels, lines))
unique_lines = [legend_dict[x] for x in unique_labels]

fig.legend(unique_lines, unique_labels, scatterpoints=1)
fig.tight_layout()
fig_name = "use_examples.png"
fig.savefig(join(figure_output_path, fig_name))
plt.close(fig)

# Plot water right location by sector
for sector in SECTOR_COLORS.keys():
    fig, ax = plt.subplots()
    df[df.sector == sector].plot(ax=ax, color=SECTOR_COLORS[sector], markersize=80, label=sector)
    crb.plot(ax=ax, zorder=-1, edgecolor="black", facecolor="none")
    flowline.plot(ax=ax, zorder=-1, edgecolor="lightblue", alpha=0.7)
    ax.set_axis_off()
    
    fig.tight_layout()
    fig_name = f"right_locations_{sector}.png"
    fig.savefig(join(figure_output_path, fig_name))
    plt.close(fig)

# Plot water right seniority histograms
for sector in SECTOR_COLORS.keys():
    fig, ax = plt.subplots()
    sns.histplot(data=df,ax=ax, x="seniority", hue="sector", hue_order=[sector], alpha=1, palette=SECTOR_COLORS)
    ax.get_legend().remove()
    
    ax.set_xlabel("Priority Date")
    ax.set_ylabel("Number of rights")
    
    fig.tight_layout()
    fig_name = f"right_priority_{sector}.png"
    fig.savefig(join(figure_output_path, fig_name))
    plt.close(fig)

# Plot water right size histograms by sector
for sector in SECTOR_COLORS.keys():
    fig, ax = plt.subplots()
    sns.histplot(data=df,ax=ax, x="log_allotment", hue="sector", hue_order=[sector], alpha=1, palette=SECTOR_COLORS)
    # ax.set_title(f"{sector}")
    ax.get_legend().remove()
    ax.set_ylabel("Number of rights")
    ax.set_xlabel("Volume log(acre-feet)")
    
    fig.tight_layout()
    fig_name = f"right_allotment_{sector}.png"
    fig.savefig(join(figure_output_path, fig_name))
    plt.close(fig)

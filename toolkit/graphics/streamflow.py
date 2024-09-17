import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import calendar


colors = [(1, 0, 0), (0, 191 / 255, 1)]  # first color is black, last is red
BR_CMAP = LinearSegmentedColormap.from_list("Custom", colors, N=100)


def plot_raster_hydrograph(streamflow):
    # if ax is None:
    #     fig, ax = plt.subplots()
    fig, ax = plt.subplots()

    raster_data = streamflow.monthly_sf_raw_2D.T
    # log transform
    raster_data = np.log1p(raster_data)
    im = ax.imshow(raster_data, cmap=BR_CMAP)
    fig.colorbar(im, shrink=0.3)
    ax.set_xlabel("Month")
    ax.set_ylabel("Gage Sites")
    tick_positions = np.arange(0, streamflow.monthly_sf_df.shape[0], 12)
    month_labels = [f"{date.year}" for date in streamflow.monthly_sf_df.index]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(month_labels[::12], rotation=45)
    ax.set_title("Log streamflows by month and site")
    return ax


def plot_raster_hydrograph_outflow(streamflow, gage_name=None):
    if gage_name is None:
        gage_name = streamflow.outflow_name

    gage_index = streamflow.gage_names.index(gage_name)
    gage_raw_monthly_sf = streamflow.monthly_sf_raw_2D[:, gage_index]
    raster_data = gage_raw_monthly_sf
    fig, ax = plt.subplots()
    raster_data = raster_data.reshape(-1, 12).T
    # log transform
    raster_data = np.log1p(raster_data)
    im = ax.imshow(raster_data, cmap=BR_CMAP)
    fig.colorbar(im)
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    xtick_positions = np.arange(0, int(streamflow.monthly_sf_df.shape[0] / 12), 5)
    ytick_positions = np.arange(0, 12)
    year_labels = list(streamflow.monthly_sf_df.index.year.unique())
    month_labels = list(streamflow.monthly_sf_df.index.month.unique())
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(year_labels[::5], rotation=45)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(month_labels)
    ax.set_title(f"Log streamflows at site {streamflow.outflow_name} by year and month")
    return ax


def plot_gage_correlation(streamflow: np.ndarray):
    correlation_matrix = np.corrcoef(streamflow)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    sns.set_style("darkgrid")
    im = ax.matshow(correlation_matrix, cmap=cmap)
    fig.colorbar(im)
    # sm = matplotlib.cm.ScalarMappable(cmap=cmap)
    # sm.set_array([np.min(correlation_matrix),np.max(correlation_matrix)])
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylabel("Basin Node", fontsize=16)
    ax.set_xlabel("Basin Node", fontsize=16)
    ax.set_title("Correlation coefficient by gage site on annual streamflow")
    return ax


def plot_compare_streamflows_correlation(
    streamflow_a: np.ndarray, streamflow_b: np.ndarray
):
    cmap = plt.get_cmap("viridis")
    sns.set_style("darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.matshow(np.corrcoef(np.transpose(streamflow_a)), cmap=cmap)
    sm = ScalarMappable(cmap=cmap)
    sm.set_array(
        [
            np.min(np.corrcoef(np.transpose(streamflow_a))),
            np.max(np.corrcoef(np.transpose(streamflow_a))),
        ]
    )
    ax.set_title("Historical Spatial Correlation", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylabel("Basin Node", fontsize=16)

    ax = fig.add_subplot(1, 2, 2)
    ax.matshow(np.corrcoef(np.transpose(streamflow_b)), cmap=cmap)
    sm = ScalarMappable(cmap=cmap)
    sm.set_array(
        [
            np.min(np.corrcoef(np.transpose(streamflow_b))),
            np.max(np.corrcoef(np.transpose(streamflow_b))),
        ]
    )
    ax.set_title("Synthetic Spatial Correlation", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylabel("Basin Node", fontsize=16)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    fig.axes[-1].set_ylabel("Pearson Correlation Coefficient", fontsize=16)
    cbar_ax.tick_params(labelsize=14)
    fig.set_size_inches(14, 6.5)
    return fig, ax

def plot_compare_streamflows_median(
    ensemble_streamflows: ndarray, historical_df: DataFrame, gage_index: int
):
    month_labels = [month[:3].upper() for month in calendar.month_name[1:]]
    gage_data=ensemble_streamflows[:,:,gage_index]
    gage_data = gage_data.reshape((gage_data.shape[0], -1, 12))
    gage_data = np.median(gage_data, axis=1)
    ensemble_df = pd.DataFrame(columns=month_labels, data=gage_data)
    ensemble_df = ensemble_df.melt()
    historical_data = historical_df.values[:, gage_index]
    historical_data = historical_data.reshape(-1, 12)
    historical_data = np.median(historical_data, axis=0)
    historical_df = pd.DataFrame(columns=month_labels, data=[historical_data])
    # ensemble_df["model"] = "Synthetic"
    fig, ax = plt.subplots()
    # historic and synthetic
    sns.boxplot(
        ax=ax,
        x=ensemble_df["variable"],
        y=ensemble_df["value"],
        # hue=ensemble_df["model"],
        showfliers=False,
    )
    ax.set(xlabel="Month", ylabel="Median (acft)")
    ax.scatter(
        x=historical_df.columns,
        y=historical_df,
        color="purple",
        marker="^",
        s=50,
        label="Historic",
        zorder=50,
    )
    ax.legend()
    return ax

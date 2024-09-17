import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import random
from os.path import join
import os
import calendar
import pandas as pd
import seaborn as sns
import geopandas as gpd
from toolkit import repo_data_path
from toolkit.utils.io import load_crb_shape, load_right_latlongs
from toolkit.graphics.palette import SECTOR_COLORS

import esda
import pysal.lib as ps

def plot_test_train_curves(train_losses, validation_losses, checkpoint_index=None):
    fig, ax = plt.subplots()
    
    ax.plot(train_losses, color="blue", label="train")
    ax.plot(validation_losses, color="orange", label="validation")
    if checkpoint_index:
        ax.axvline(x=checkpoint_index, linestyle="--", color="black", label="selected timepoint")

    plt.legend()
    return fig, ax


def visualize_predictions(actual, predicted, sample_ids=-1):
    """
    Plots actual vs. predicted allocations for a given sample index.

    Parameters:
    - actual (torch.Tensor): The true allocations tensor.
    - predicted (torch.Tensor): The predicted allocations tensor.
    - sample_idx (int): Index of the sample to visualize.
    """
    if sample_ids == -1:
        sample_ids = range(actual.shape[0])
    fig, ax = plt.subplots()
    for idx in sample_ids:
        # Extract data for the given sample index
        actual_data = actual[idx]
        predicted_data = predicted[idx]

        # Plot the actual and predicted data
        ax.plot(actual_data, label="Actual", color="blue")
        ax.plot(predicted_data, label="Predicted", color="red", linestyle="dashed")

    # Formatting
    ax.set_title(f"Actual vs. Predicted Allocations for Sample {idx}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Allocation Value")
    # fig.legend()
    return fig, ax

def plot_streamflow(array, log=True, xlabels=None, y_labels=None):
    fig, ax = plt.subplots()
    if log:
        array = np.log1p(array)
    
    ims = []
    ims.append(ax.imshow(array))
    
    # axis
    ax.set_xlabel("Gage site")
    ax.set_ylabel("Month")
    # ax.set_xtick_labels()
    # ax.set_ytick_labels()
    
    ax.set_title("Input Streamflow")

    return fig, ax

def plot_compare_rasters(target_array, predicted_array):
    global_min = np.min([target_array.min(), predicted_array.min()])
    global_max = np.max([target_array, predicted_array])
    
    fig, ax = plt.subplots(2)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    
    ims = []
    ims.append(ax[0].imshow(target_array))
    ims.append(ax[1].imshow(predicted_array))
    
    # axes labels
    for a in ax:
        a.set_xlabel("Right")
        a.set_ylabel("Month")
        
    # titles
    ax[0].set_title("WRAP Right Shortages")
    ax[1].set_title("Emulator Predicted Right Shortages")
    fig.colorbar(ims[0], ax=cbar_ax, fraction=0.046, pad=0.04, shrink=0.5)
    
    return fig, ax


def plot_pointwise_mse(target_array, predicted_array):
    # computations
    se = np.sqrt((target_array - predicted_array)**2)
    
    # figure creation
    fig, ax = plt.subplots()
    ims = []
    ims.append(ax.imshow(se, vmin=0, vmax=1))
    
    # axes labels
    ax.set_xlabel("Right")
    ax.set_ylabel("Month")
    
    # title
    ax.set_title("Pointwise Squared-Error")
    fig.colorbar(ims[0], fraction=0.046, pad=0.04, shrink=0.5)

    return fig, ax


def plot_right_timeseries(target_array, predicted_array, right_index=None):
    if right_index is None:
        right_index = random.randint(0, target_array.shape[1]) # get random right to plot

    # generate figure
    fig, ax = plt.subplots()
    ax.plot(target_array[:,right_index], label="Actual", color="blue")
    ax.plot(predicted_array[:,right_index], label="Predicted", color="red")

    # Formatting
    ax.set_title(f"Actual vs. Predicted Shortage Ratios for {right_index}th right")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Shortage Ratio")
    fig.legend()

    return fig, ax


def plot_error_histograms(target_array, predicted_array, right_index=None, show=True, file_name=None):
        
    rse = np.sqrt((target_array - predicted_array)**2) # shape (months, rights)
    yearly_rmse = rse.reshape((-1, 12, rse.shape[1])) # shape (year, month, right)
    
    # sanity check of re-indexing
    assert (rse[0] == yearly_rmse[0,0]).all()
    assert (rse[1] == yearly_rmse[0,1]).all()
    assert (rse[12] == yearly_rmse[1,0]).all()
    
    yearly_rmse = yearly_rmse.mean(axis=1) # shape (years, rights)
    fig, ax = plt.subplots()
    ax.boxplot(yearly_rmse.T)
    ax.set_title("Boxplots of errors across years")
    ax.set_ylabel("Size of Error")
    ax.set_xlabel("Year (2005-2016)")
        
    if file_name:
        fig.savefig(file_name)
    return fig, ax


def visualize_geospatial(target_array, predicted_array, right_labels, error_type="rmse"):
    if error_type == "rmse":
        error = np.mean((target_array - predicted_array)**2, axis=1)
        
    elif error_type == "nse":
        error = np.sum((target_array - predicted_array)**2, axis=1, keepdims=True)
        true_variances = np.sum((target_array - np.mean(target_array, axis=1, keepdims=True))**2, axis=1,keepdims=True)
        true_variances[true_variances < 0.01] = 0.0
        error = 1 - (error / true_variances)
        error[np.isinf(error)] = np.nan
    
    error = np.nanmean(error,axis=0)
        
    right_gdf = gpd.GeoDataFrame(index=right_labels, data=error.flatten(), columns=["error"])
    latlongs = load_right_latlongs()
    latlongs["error"] = right_gdf["error"]
    latlongs = latlongs.loc[latlongs.index.intersection(right_gdf.index)]
    
    fig, ax = plt.subplots()

    if error_type=="rmse":
        latlongs.plot(column="error", legend=True, alpha=0.5, missing_kwds={"color": "red"}, ax=ax)
    elif error_type=="nse":
        # vmin = latlongs.error.min()
        # vmax = latlongs.error.max()
        latlongs[latlongs.error > 0.0].plot(column="error", legend=True, alpha=0.5, ax=ax)
        if latlongs[latlongs.error <= 0.0].size > 0:
            latlongs[latlongs.error <= 0.0].plot(ax=ax, color="red", legend=True, alpha=0.5)

    # add crb outline
    crb = load_crb_shape()
    crb.to_crs(latlongs.crs, inplace=True)
    crb.plot(ax=ax, zorder=-1, edgecolor="black", facecolor="none")
    return fig, ax, latlongs


def plot_temporal_boxplot(target_array, predicted_array, index=None, axis="year"):
    if index is not None:
        target_array = target_array[index]
        predicted_array = predicted_array[index]
        rmse = np.sqrt((target_array - predicted_array)**2) # shape (months, rights)
        yearly_rmse = rmse.reshape((-1, 12, rmse.shape[1])) # shape (year, month, right)
        
        # sanity check of re-indexing
        assert (rmse[0] == yearly_rmse[0,0]).all()
        assert (rmse[1] == yearly_rmse[0,1]).all()
        assert (rmse[12] == yearly_rmse[1,0]).all()
        
        yearly_rmse = yearly_rmse.mean(axis=1) # shape (years, rights)
        
        fig, ax = plt.subplots()
        ax.boxplot(yearly_rmse.T)
        ax.set_title("Boxplots of errors across years")
        ax.set_ylabel("Size of Error")
        ax.set_xlabel("Year (2005-2016)")
    else:
        rmse = np.sqrt((target_array - predicted_array)**2) # shape (samples, months, rights)
        yearly_rmse = rmse.reshape((-1, int(rmse.shape[1]/ 12), 12, rmse.shape[2])) # shape (samples, years, month, right)
        
        # sanity check of re-indexing
        assert (rmse[0,0] == yearly_rmse[0,0,0]).all()
        assert (rmse[0,1] == yearly_rmse[0,0,1]).all()
        assert (rmse[0,12] == yearly_rmse[0,1,0]).all()
        if axis == "year":
            yearly_rmse = yearly_rmse.mean(axis=0) # average rmse across datapoints -> shape (years, month, right)
            yearly_rmse = yearly_rmse.mean(axis=1) # average rmse across months -> shape (years, rights)
        elif axis == "right":
            yearly_rmse = yearly_rmse.mean(axis=1) # average rmse across years -> shape (samples, month, right)
            yearly_rmse = yearly_rmse.mean(axis=1) # average rmse across months -> shape (samples, rights)
            yearly_rmse = yearly_rmse.swapaxes(0,1)
        fig, ax = plt.subplots()
        ax.boxplot(yearly_rmse.T)
        ax.set_title("Boxplots of errors across years")
        ax.set_ylabel("Size of Error")
        ax.set_xlabel("Year (2005-2016)")
        
    return fig, ax



def evaluate_sectoral_error(true_array, pred_array, sectors):
    mse = np.sqrt((true_array - pred_array)**2).mean(axis=1) # mean squared error averaged along time axis
    sector_list = []
    sector_mse_list = []
    for sector in np.unique(sectors):
        sector_mask = sectors == sector
        sector_mse = mse[:,sector_mask].flatten()
        sector_list.append(sector)
        sector_mse_list.append(sector_mse)
        

    # create sns compatible dataframe
    data_df = pd.DataFrame(index=sector_list, data=sector_mse_list).T.melt()
    
    fig, ax = plt.subplots()
    # historic and synthetic
    sns.boxplot(
        ax=ax,
        x="variable",
        y="value",
        data=data_df,
        # hue=ensemble_df["model"], 
        showfliers=False,
    )
    return fig, ax


def evaluate_seniority_error(true_array, pred_array, seniority):
    mse = np.sqrt((true_array - pred_array)**2).mean(axis=1) # mean squared error averaged along time axis
    
    # convert seniority to datetimes
    

    # create sns compatible dataframe
    data_df = pd.DataFrame(columns=seniority, data=mse).T
    data_df.index = data_df.index.to_datetime()
    
    # sort index
    data_df = data_df.sort_index()
    
    # plot scatter
    fig, ax = plt.subplots()
    ax.scatter(x=data_df.index, y=data_df.mean(axis=1))
    
    # noodle with index labels
    
    return fig, ax


def evaluate_sectoral_variance(true_array, sectors):
    variance = np.var(true_array, axis=1) # mean squared error averaged along time axis
    sector_list = []
    sector_var_list = []
    for sector in np.unique(sectors):
        sector_mask = sectors == sector
        sector_var = variance[:,sector_mask].flatten()
        sector_list.append(sector)
        sector_var_list.append(sector_var)
    
    # create sns compatible dataframe
    data_df = pd.DataFrame(index=sector_list, data=sector_var_list).T.melt()
    
    fig, ax = plt.subplots()
    # historic and synthetic
    sns.boxplot(
        ax=ax,
        x="variable",
        y="value",
        data=data_df,
        # hue=ensemble_df["model"], 
        showfliers=False,
    )

    return fig, ax


def evaluate_sectoral_bins(true_array, sectors):
    
    # compute the average amount a right timeseries has value 1, 0, or something in between
    is_one = (true_array == 1.0).mean(axis=1)
    is_zero = (true_array == 0.0).mean(axis=1)
    middle = ~((true_array == 1.0)| (true_array == 0.0)).mean(axis=1) # double check this logic
    sector_list = []
    sector_var_list = []
    
    masks = [is_one, is_zero, middle]
    mask_names = ["one", "zero", "middle"]
    
    fig, axes = plt.subplots(len(masks))
    for i in range(len(masks)):
        ax = axes[i]
        mask = masks[i]
        for sector in np.unique(sectors):
            sector_mask = sectors == sector
            sector_var = mask[:,sector_mask,:].flatten()
            sector_list.append(sector)
            sector_var_list.append(sector_var)
        
        # create sns compatible dataframe
        data_df = pd.DataFrame(index=sector_list, data=sector_var_list).T.melt()
        
        # historic and synthetic
        sns.boxplot(
            ax=ax,
            x="variable",
            y="value",
            data=data_df,
            # hue=ensemble_df["model"], 
            showfliers=False,
        )
        ax.set_title(mask_names[i])

    return fig, ax

def generate_overall_metrics(results_tuple, right_labels):
    """generates average MAE, ME, RMSE, NSE for ratios and volume for the given dataset

    Parameters
    ----------
    results_tuple : tuple(np.array)
        (inputs, outputs, predictions)
    right_labels : list[str] 
        labels for rights included in the data

    Returns
    -------
    dict
    """
    
    variance_threshold = 0.01
    
    # init metrics dictionary
    metrics_dict = {}
    volumetric_metrics_dict = {}
    
    # unpack results
    inputs, true_outputs, pred_outputs = results_tuple
    
    # compute mse
    metrics_dict["overall_mse"] = np.mean((true_outputs - pred_outputs)**2, axis=(1,2))
    
    # compute mae
    metrics_dict["overall_mae"] = np.mean(abs(true_outputs - pred_outputs), axis=(1,2))
    
    # compute me
    metrics_dict["overall_me"] = np.mean(true_outputs - pred_outputs, axis=(1,2))
    
    # compute nse
    squared_error = np.sum((true_outputs-pred_outputs)**2, axis=1)
    squared_deviation_from_mean = np.sum((true_outputs-np.mean(true_outputs,keepdims=True,axis=1))**2, axis=1)
    # squared_deviation_from_mean[squared_deviation_from_mean < variance_threshold] = 0.0
    nse = 1-(squared_error/squared_deviation_from_mean)
    nse[nse == -np.inf] = np.nan
    nse[nse < -1] = -1
    
    metrics_dict["overall_nse"] = np.nanmean(nse, axis=1)
    
    # compute volumetric values
    historical_diversions = pd.read_csv(join(repo_data_path, "colorado-full", "C3_diversions.csv"))
    historical_diversions["date"] = pd.to_datetime(historical_diversions[["year", "month"]].assign(DAY=1))

    allotments = historical_diversions.pivot_table(
        index="date",
        columns="water_right_identifier",
        values="diversion_or_energy_target",
        dropna=False
    )
    monthly_targets = allotments.iloc[0:12]
    monthly_targets = monthly_targets.loc[:,right_labels]
    
    n_years = true_outputs.shape[1] // 12
    monthly_targets_repeat = np.tile(monthly_targets.values,(n_years,1))
    size_adjusted_true_outputs = true_outputs * monthly_targets_repeat
    size_adjusted_pred_outputs = pred_outputs * monthly_targets_repeat
    
    # volumetric rmse
    volumetric_metrics_dict["overall_volumetric_mse"] = np.mean((size_adjusted_true_outputs - size_adjusted_pred_outputs)**2, axis=(1,2))
    
    # volumetric mae
    volumetric_metrics_dict["overall_volumetric_mae"] = np.mean(abs(size_adjusted_true_outputs - size_adjusted_pred_outputs), axis=(1,2))
    
    # volumetric me
    volumetric_metrics_dict["overall_volumetric_me"] = np.mean(size_adjusted_true_outputs - size_adjusted_pred_outputs, axis=(1,2))
    
    # compute nse
    squared_error = np.sum((size_adjusted_true_outputs-size_adjusted_pred_outputs)**2, axis=1)
    squared_deviation_from_mean = np.sum((size_adjusted_true_outputs-np.mean(size_adjusted_true_outputs,keepdims=True,axis=1))**2, axis=1)
    
    # round low variance time series to 0 for later filtering using nanmean
    nse = 1-(squared_error/squared_deviation_from_mean)
    nse[nse == -np.inf] = np.nan
    
    # round very low NSE up to -1
    nse[nse < -1] = -1
    
    volumetric_metrics_dict["overall_volumetric_nse"] = np.nanmean(nse, axis=1)
    
    return metrics_dict, volumetric_metrics_dict

def process_streamflow(results_tuple, aggregation=None):
    inputs, true_outputs, pred_outputs = results_tuple
    
    # extract streamflow
    outflow_index = 42 # index of k20000 gage site
    sf_data = inputs[:, :, outflow_index].squeeze()
    if aggregation=="years":
        sf_data = sf_data.reshape((sf_data.shape[0], -1, 12))
        sf_data = sf_data.mean(axis=1)
    elif aggregation=="points":
        sf_data = sf_data.reshape((sf_data.shape[0], -1, 12))
        sf_data = sf_data.mean(axis=0)
    else:
        sf_data = sf_data.reshape((-1, 12))
    
    month_labels = range(1,13)
    streamflow_df = pd.DataFrame(columns=month_labels, data=sf_data)
    streamflow_df = streamflow_df.melt()
    
    return streamflow_df

def process_shortage(results_tuple, aggregation=None):
    inputs, true_outputs, pred_outputs = results_tuple

    shortage_data = true_outputs.mean(axis=2) # mean over rights
    
    if aggregation=="years":
        shortage_data = shortage_data.reshape((shortage_data.shape[0], -1, 12))
        shortage_data = shortage_data.mean(axis=1)
    elif aggregation=="points":
        shortage_data = shortage_data.reshape((shortage_data.shape[0], -1, 12))
        shortage_data = shortage_data.mean(axis=0)
    else:
        shortage_data = shortage_data.reshape((-1, 12))
    
    month_labels = range(1,13)
    shortage_df = pd.DataFrame(columns=month_labels, data=shortage_data)
    shortage_df = shortage_df.melt()
    
    return shortage_df

def process_errors(
    results_tuple,
    right_labels,
    sector,
    priority_date,
    allotment,
    time_index):
    # unpack results
    inputs, true_outputs, pred_outputs = results_tuple

    # create right dataframe
    rights_df = gpd.GeoDataFrame(index=right_labels)
    
    # latlong
    latlongs = load_right_latlongs()
    latlongs = latlongs.loc[rights_df.index]
    rights_df["geometry"] = latlongs.geometry
    rights_df.set_geometry("geometry")
    
    # metadata
    rights_df["sector"] = sector
    rights_df["priority_date"] = pd.to_datetime(priority_date)
    rights_df["ordinal_seniority"] = rights_df["priority_date"].apply(lambda x: x.toordinal())
    rights_df["allotment"] = allotment
    rights_df["log_allotment"] = np.log10(rights_df.allotment)
    
    # define right allotment categories
    cutoff = 0.9
    rights_df["large_right"] = rights_df.allotment > rights_df.allotment.quantile(cutoff)
    rights_df["small_right"] = ~rights_df["large_right"]
    print("Total annual allotment of top 10%: ", rights_df[rights_df.large_right].allotment.sum() / rights_df.allotment.sum())
    print("Total annual allotment of top 10%: ", rights_df[rights_df.small_right].allotment.sum() / rights_df.allotment.sum())
    
    # create temporal dataframe
    time_df = pd.DataFrame(index=time_index)
    time_df["year"] = time_df.index.year
    time_df["month"] = time_df.index.month
    
    #load in historical diversion output to extract monthly targets 
    historical_diversions = pd.read_csv(join(repo_data_path, "colorado-full", "C3_diversions.csv"))
    historical_diversions["date"] = pd.to_datetime(historical_diversions[["year", "month"]].assign(DAY=1))
    allotments = historical_diversions.pivot_table(
        index="date",
        columns="water_right_identifier",
        values="diversion_or_energy_target",
        dropna=False
    )
    monthly_targets = allotments.iloc[0:12]
    monthly_targets = monthly_targets.loc[:,right_labels]
        
    ### error metrics
    # MSE
    rights_df["mse"] = np.mean((true_outputs - pred_outputs)**2, axis=(0,1))
    time_df["mse"] = np.mean((true_outputs - pred_outputs)**2, axis=(0,2))

    # MAE
    rights_df["mae"] = np.mean(abs(true_outputs - pred_outputs), axis=(0,1))
    time_df["mae"] = np.mean(abs(true_outputs - pred_outputs), axis=(0,2))

    # ME
    rights_df["me"] = np.mean(true_outputs - pred_outputs, axis=(0,1))
    time_df["me"] = np.mean(true_outputs - pred_outputs, axis=(0,2))
    
    # NSE
    # df["nse"] = np.mean(1-(np.sum((true_outputs-pred_outputs)**2, axis=1)/np.sum((true_outputs-np.mean(true_outputs))**2, axis=1)),axis=0)
    squared_error = np.sum((true_outputs-pred_outputs)**2, axis=1)
    squared_deviation_from_mean = np.sum((true_outputs-np.mean(true_outputs,keepdims=True,axis=1))**2, axis=1)
    nse = 1-(squared_error/squared_deviation_from_mean)
    nse[nse == -np.inf] = np.nan
    nse[nse < -1] = -1 # cutoff at -1 to account for extremely large, negative errors
    rights_df["nse"] = np.nanmean(nse, axis=0)
    
    # volumetric versions of error metrics
    n_years = true_outputs.shape[1] // 12
    monthly_targets_repeat = np.tile(monthly_targets.values,(n_years,1))
    size_adjusted_true_outputs = true_outputs * monthly_targets_repeat
    size_adjusted_pred_outputs = pred_outputs * monthly_targets_repeat
    
    # volumetric mse
    rights_df["volumetric_mse"] = np.mean((size_adjusted_true_outputs - size_adjusted_pred_outputs)**2, axis=(0,1))
    time_df["volumetric_mse"] = np.mean((size_adjusted_true_outputs - size_adjusted_pred_outputs)**2, axis=(0,2))

    # volumetric mae
    rights_df["volumetric_mae"] = np.mean(abs(size_adjusted_true_outputs - size_adjusted_pred_outputs), axis=(0,1))
    time_df["volumetric_mae"] = np.mean(abs(size_adjusted_true_outputs - size_adjusted_pred_outputs), axis=(0,2))

    # volumetric me
    rights_df["volumetric_me"] = np.mean(size_adjusted_true_outputs - size_adjusted_pred_outputs, axis=(0,1))
    time_df["volumetric_me"] = np.mean((size_adjusted_true_outputs - size_adjusted_pred_outputs), axis=(0,2))

    # volumetric nse
    squared_error = np.sum((size_adjusted_true_outputs-size_adjusted_pred_outputs)**2, axis=1)
    squared_deviation_from_mean = np.sum((size_adjusted_true_outputs-np.mean(size_adjusted_true_outputs,keepdims=True,axis=1))**2, axis=1)
    nse = 1-(squared_error/squared_deviation_from_mean)
    nse[nse == -np.inf] = np.nan # process nans
    nse[nse < -1] = -1 # cutoff again
    rights_df["volumetric_nse"] = np.nanmean(nse, axis=0)
    
    # variance
    rights_df["variance"] = np.var(true_outputs,axis=1).mean(axis=0)
    # df["variance"] = np.sum((true_outputs-np.mean(true_outputs,axis=1,keepdims=True))**2,axis=1) / true_outputs.shape[1]
    
    # aggregate metrics for overall evaluation of error
    overall_df = pd.DataFrame(columns=["MSE", "MAE", "ME", "NSE"])
    large_rights_df = rights_df.loc[rights_df.large_right]
    small_rights_df = rights_df.loc[rights_df.small_right]
    
    #ratio
    overall_df.loc["All rights ratio",:] = [
        rights_df["mse"].mean(),
        rights_df["mae"].mean(), 
        rights_df["me"].mean(), 
        rights_df["nse"].mean()]
    
    overall_df.loc["Large rights ratio",:] = [
        large_rights_df["mse"].mean(),
        large_rights_df["mae"].mean(), 
        large_rights_df["me"].mean(), 
        large_rights_df["nse"].mean()]
    
    overall_df.loc["Small rights ratio",:] = [
        rights_df["mse"].mean(),
        rights_df["mae"].mean(), 
        rights_df["me"].mean(), 
        rights_df["nse"].mean()]
    
    #volume
    
    overall_df.loc["All rights volume",:] = [
        rights_df["volumetric_mse"].mean(),
        rights_df["volumetric_mae"].mean(), 
        rights_df["volumetric_me"].mean(), 
        rights_df["volumetric_nse"].mean()]
    
    overall_df.loc["Large rights volume",:] = [
        large_rights_df["volumetric_mse"].mean(),
        large_rights_df["volumetric_mae"].mean(), 
        large_rights_df["volumetric_me"].mean(), 
        large_rights_df["volumetric_nse"].mean()]

    overall_df.loc["Small rights volume",:] = [
        small_rights_df["volumetric_mse"].mean(),
        small_rights_df["volumetric_mae"].mean(), 
        small_rights_df["volumetric_me"].mean(), 
        small_rights_df["volumetric_nse"].mean()]
    
    overall_df = overall_df.astype(float)
    overall_df = overall_df.round(3)
    
    # correlations
    
       
    ### correlations
    correlation_df = pd.DataFrame(columns=["stat", "pvalue"])
    
    # PCC for priority date
    metrics = ["volumetric_mse", "volumetric_mae", "volumetric_me", "volumetric_nse", "mse", "mae", "me", "nse"]
    priority_corr_df = pd.DataFrame(index=metrics, columns=["stat", "pvalue"])
    
    for metric in metrics:
        # filter out nans
        metric_column = rights_df[metric].dropna()
        metadata_column = rights_df.loc[metric_column.index,"ordinal_seniority"]
        
        # normalize
        z = (metric_column - np.mean(metric_column)) / np.std(metric_column)
        z2 = (metadata_column - np.mean(metadata_column)) / np.std(metadata_column)
        
        # compute and store
        stat, pvalue = sp.stats.spearmanr(z, z2)
        priority_corr_df.loc[metric,"stat"] = stat
        priority_corr_df.loc[metric,"pvalue"] = pvalue
    
    # PCC for total allotment size
    metrics = ["volumetric_mse", "volumetric_mae", "volumetric_me", "volumetric_nse", "mse", "mae", "me", "nse"]
    allotment_corr_df = pd.DataFrame(index=metrics, columns=["stat", "pvalue"])
    
    for metric in metrics:
        # filter out nans
        metric_column = rights_df[metric].dropna()
        metadata_column = rights_df.loc[metric_column.index,"allotment"]
        
        # normalize
        z = (metric_column - np.mean(metric_column)) / np.std(metric_column)
        z2 = (metadata_column - np.mean(metadata_column)) / np.std(metadata_column)
        
        
        # compute and save
        stat, pvalue = sp.stats.spearmanr(z, z2)
        allotment_corr_df.loc[metric,"stat"] = stat
        allotment_corr_df.loc[metric,"pvalue"] = pvalue
        
    # Spatial autocorrelation with Moran's I
    
    # distance matrix
    def compute_distance(gdf, k=10):
        w = ps.weights.KNN.from_dataframe(gdf, k=k)
        w.transform = 'R'
        return w
        
    spatial_corr_df = pd.DataFrame(index=metrics, columns=["stat", "pvalue"])
    # rights_df["logvolmse"] = np.log(rights_df.volumetric_mse)
    metrics = ["volumetric_mse", "volumetric_mae", "volumetric_me", "volumetric_nse", "mse", "mae", "me", "nse"]
    for metric in metrics:        
        metric_column = rights_df[metric].dropna()
        
        w = compute_distance(rights_df.loc[metric_column.index], k=5)
        
        # normalize
        z = (metric_column - np.mean(metric_column)) / np.std(metric_column)        

        # Calculate Local Moran's I
        z = z.astype(np.float64)  # Ensure standardized residuals are float64

        mi = esda.Moran(z, w)
        local_mi = esda.Moran_Local(z, w)
        
        # save local results to rights df
        rights_df.loc[metric_column.index, f'{metric}_local_moran_I'] = local_mi.Is
        rights_df.loc[metric_column.index, f'{metric}_local_moran_p'] = local_mi.p_sim
        
        # save global results to corr df
        stat, pvalue = (mi.I, mi.p_norm)
        spatial_corr_df.loc[metric,"stat"] = stat
        spatial_corr_df.loc[metric,"pvalue"] = pvalue
    
    corr_dict = {"spatial_corr_df": spatial_corr_df, "allotment_corr_df": allotment_corr_df, "priority_corr_df": priority_corr_df}
    
    error_dict = {"rights_df": rights_df, "time_df": time_df, "overall_df": overall_df, "corr_dict": corr_dict}
    
    return error_dict

def generate_results(
    errors_dict,
    streamflow_df,
    figure_folder):
    # unpack values
    overall_df = errors_dict["overall_df"]
    time_df = errors_dict["time_df"]
    rights_df = errors_dict["rights_df"]
    corr_dict = errors_dict["corr_dict"]
    spatial_corr_df = corr_dict["spatial_corr_df"]
    allotment_corr_df = corr_dict["allotment_corr_df"]
    priority_corr_df = corr_dict["priority_corr_df"]
    
    
    os.makedirs(join(figure_folder, "error_breakdowns"), exist_ok=True)
    
    # load in geospatial data
    # load in crb
    crb_gdf_path = join(repo_data_path, "geospatial", "CRB")
    crb = gpd.read_file(crb_gdf_path)
    crb.to_crs(rights_df.crs, inplace=True)

    # load in flowlines
    flowline_gdf_path = join(repo_data_path, "geospatial", "Flowline")
    flowline = gpd.read_file(flowline_gdf_path)
    flowline.to_crs(rights_df.crs, inplace=True)
    
    ### tables
    def round_to_3(x):
        return f"{x:.3f}"
    
    rename_dict = {
        "volumetric_mse": "Volume MSE", 
        "volumetric_mae": "Volume MAE", 
        "volumetric_me": "Volume ME", 
        "volumetric_nse": "Volume NSE", 
        "mse": "MSE", 
        "mae": "MAE", 
        "me": "ME", 
        "nse": "NSE",
    }
    

    # Create a dictionary of formatters
    
    ## overall results
    formatters = {column: round_to_3 for column in overall_df.columns}
    overall_df.to_latex(join(figure_folder, "error_summary_df.tex"), formatters=formatters)
    
    ## spatial correlations
    formatters = {column: round_to_3 for column in spatial_corr_df.columns}
    spatial_corr_df.rename(rename_dict).astype(float).round(3).to_latex(join(figure_folder, "spatial_corr_df.tex"), formatters=formatters)
    
    ## metadata correlations
    formatters = {column: round_to_3 for column in allotment_corr_df.columns}
    allotment_corr_df.rename(rename_dict).to_latex(join(figure_folder, "allotment_corr_df.tex"), formatters=formatters)
    
    formatters = {column: round_to_3 for column in priority_corr_df.columns}
    priority_corr_df.rename(rename_dict).to_latex(join(figure_folder, "priority_corr_df.tex"), formatters=formatters)
    
    ### plots
    ## Geospatial error plots
    metrics = ["volumetric_mse", "volumetric_mae", "volumetric_me", "volumetric_nse", "mse", "mae", "me", "nse"]
    metric_names = [r"MSE $(\\text{acre-feet}^2)$", "MAE (acre-feet)", "ME (acre-feet)", "NSE",
                    r"MSE $(\\text{ratio}^2)$", "MAE (ratio)", "ME (ratio)", "NSE"]
    # metric_names = ["MSE", "MAE (acre-feet)", "ME (acre-feet)", "NSE",
    #                 "MSE", "MAE (ratio)", "ME (ratio)", "NSE"]
    for i in range(len(metrics)):
        fig, ax = plt.subplots()
        metric_column = metrics[i]
        
        rights_df.plot(ax=ax, column=metric_column, cmap="plasma", markersize=80, alpha=0.75, legend=True, legend_kwds={'label': f"{metric_names[i]}"})
        crb.plot(ax=ax, zorder=-1, edgecolor="black", linewidth=2, facecolor="none")
        flowline.plot(ax=ax, zorder=-1, edgecolor="grey", alpha=0.5)
        
        fig.tight_layout()
        fig_name  = f"geospatial_{metric_column}.png"
        fig.savefig(join(figure_folder, "error_breakdowns", fig_name))
        plt.close(fig)
    
    ## Monthly volumetric errors with streamflow
    metrics = ["volumetric_mse", "volumetric_mae", "volumetric_me", "mse", "mae", "me"]
    metric_names = [r"MSE $(\\text{acre-feet}^2)$", "MAE (acre-feet)", "ME (acre-feet)", 
                    r"MSE $(\\text{ratio}^2)$", "MAE (ratio)", "ME (ratio)"]
    
    for i in range(len(metrics)):
        fig, ax = plt.subplots(figsize=(16,7))
        metric_column = metrics[i]
        error_color = "orange"
        streamflow_color = "lightblue"
        box_width = 0.2
        
        positions = np.arange(0,12)
        positions1 = positions - 0.1
        positions2 = positions + 0.1
        
        # error
        sns.boxplot(
            ax=ax,
            x="month",
            y=metric_column,
            data=time_df,
            showfliers=False,
            color=error_color,
            width=box_width,
            positions=positions1
        )
        ax.yaxis.label.set_color(error_color)
        ax.set_ylabel(f"{metric_names[i]}")
        
        # streamflow
        ax2 = ax.twinx()
        sns.boxplot(
            ax=ax2,
            x="variable",
            y="value",
            data=streamflow_df,
            showfliers=False,
            color=streamflow_color,
            width=box_width,
            positions=positions2
        )
        
        ax2.yaxis.label.set_color(streamflow_color)
        ax2.set_ylabel("Streamflow (acre-feet)")
        ax2.set_xlabel("Month")
        ax2.set_xticklabels([mon[:3] for mon in calendar.month_name[1:]], rotation=45, ha='right')
        
        fig.tight_layout()
        fig_name  = f"monthly_{metric_column}.png"
        fig.savefig(join(figure_folder, "error_breakdowns", fig_name))
        plt.close(fig)
        
    
    ## Yearly errors
    metrics = ["volumetric_mse", "volumetric_mae", "volumetric_me", "mse", "mae", "me"]
    metric_names = [r"MSE $(\\text{acre-feet}^2)$", "MAE (acre-feet)", "ME (acre-feet)", 
                    r"MSE $(\\text{acre-feet}^2)$", "MAE", "ME"]
    for i in range(len(metrics)):
        fig, ax = plt.subplots(figsize=(16,7))
        metric_column = metrics[i]
        
        year_means = time_df.groupby("year")[metric_column].mean()
        year_stds = time_df.groupby("year")[metric_column].std()
        
        line = sns.lineplot(x=year_means.index, y=year_means,marker="o", ax=ax)
        ax.fill_between(year_means.index, year_means - year_stds, year_means + year_stds, alpha=0.3)
        
        ax.set(xlabel="Year", ylabel=metric_names[i])
        
        # ### linear regression
        # start_index = 10
        # x = year_means[start_index:].index
        # y = year_means[start_index:].values
        # slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
        # print(metric_column, slope)
        
        # y_pred = slope * x + intercept
        # ax.plot(x, y_pred, color='red', label='Regression line')
        
        # # normalized slope
        # x1_normalized = (x - np.mean(x)) / np.std(x)
        # y1_normalized = (y - np.mean(y)) / np.std(y)
        # slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x1_normalized,y1_normalized)
        
        # print(metric_column, "norm slope: ", slope)
        # # y_pred = slope * x + intercept
        # ax.plot(x, y_pred, color='red', label='Regression line')
        
        fig.tight_layout()
        fig_name  = f"yearly_{metric_column}.png"
        fig.savefig(join(figure_folder, "error_breakdowns", fig_name))
        plt.close(fig)
    
    ## Sectoral
    metrics = ["volumetric_mse", "volumetric_mae", "volumetric_me", "volumetric_nse", "mse", "mae", "me", "nse"]
    metric_names = [r"MSE $(\\text{acre-feet}^2)$", "MAE (acre-feet)", "ME (acre-feet)", "NSE",
                    r"MSE $(\\text{ratio}^2)$", "MAE (ratio)", "ME (ratio)", "NSE"]
    
    error_medians = dict()
    for i in range(len(metrics)):
        fig, ax = plt.subplots()
        metric_column = metrics[i]
        
        boxplot = sns.boxplot(
            ax=ax,
            x="sector",
            y=metric_column,
            data=rights_df,
            showfliers=False,
            palette=SECTOR_COLORS
        )
        
        # extract medians
        error_medians[metric_column] = dict()
        for sector in SECTOR_COLORS.keys():
            error_medians[metric_column][sector] = rights_df[rights_df.sector==sector][metric_column].median()
        
        iqrs = dict()
        for sector in SECTOR_COLORS.keys():
            Q1 = rights_df[rights_df.sector==sector][metric_column].quantile(0.25)
            Q3 = rights_df[rights_df.sector==sector][metric_column].quantile(0.75)
            IQR = Q3 - Q1
            iqrs[sector] = IQR
        
        ax.set_ylabel(metric_names[i])
        ax.set_xlabel("Sector")
        
        # fig.suptitle("Error across Sectors")
        fig.tight_layout()
        fig_name  = f"sectoral_{metric_column}.png"
        fig.savefig(join(figure_folder, "error_breakdowns", fig_name))
        ax.set_ylabel(metric_names[i])
        ax.set_xlabel("Sector")
        
        # fig.suptitle("Error across Sectors")
        fig.tight_layout()
        fig_name  = f"sectoral_{metric_column}.png"
        fig.savefig(join(figure_folder, "error_breakdowns", fig_name))
        plt.close(fig)
    
    error_medians_df = pd.DataFrame(error_medians)
    error_medians_df[["mse", "mae", "me", "nse"]].to_latex(join(figure_folder, "sector_error_medians.tex"), formatters=formatters)
    
    return
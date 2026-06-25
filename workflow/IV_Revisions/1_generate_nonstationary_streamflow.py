from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_rel
from toolkit.utils.io import hdf5_to_dict, dict_to_hdf5
from toolkit import outputs_data_path, repo_data_path
from toolkit.graphics.palette import DROUGHT_PALETTE, MEANPROPS
from toolkit.wrap.io import flo_to_df

## Settings ##
# Choose a test dataset to modify (using drought level 0.0 as baseline)
drought_level = 0.0
# Transformation parameters
wet_addition_amount = 1000000  # Total amount to add progressively (acre-feet per month)
dry_fixed_decrease = 1000000     # Fixed amount to decrease progressively (acre-feet per month)
dry_minimum_fraction = 0.25    # Minimum fraction of original value to maintain (25%)

## Path Configuration ##
# Use local outputs directory in the same folder as this script
local_outputs_path = join(os.path.dirname(__file__), "outputs")
if not os.path.exists(local_outputs_path):
    os.makedirs(local_outputs_path)

# Input dataset path (still from main outputs directory)
input_dataset_path = join(
    outputs_data_path,
    "synthetic-test", 
    f"synthetic_test_dataset_drought_{str(drought_level)}.h5"
)

# Output path for non-stationary dataset (save to local outputs)
output_dataset_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_nonstationary.h5"
)

# Figure output path (save to local outputs)
figure_output_path = join(local_outputs_path, "figures", "nonstationary_comparison")
if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)

## Main Script ##

# Load the test dataset
data_dict = hdf5_to_dict(input_dataset_path)

# Extract only the first 10 streamflows for faster processing
print("Extracting first 10 streamflows from original dataset...")
streamflow_data_full = data_dict["streamflow_data"]  # shape: (n_runs, n_timesteps, n_locations)

# n_subset = 10
n_subset = streamflow_data_full.shape[0]

streamflow_data = streamflow_data_full[:n_subset, :, :]  # Take first 10 runs
time_index = pd.to_datetime(data_dict["streamflow_index"])

print(f"Original dataset shape: {streamflow_data_full.shape}")
print(f"Subset dataset shape: {streamflow_data.shape}")

# Debug: Check for NaN values in original data
print("\n=== DEBUG: Checking for NaN values in original data ===")
original_nans = np.isnan(streamflow_data).sum()
print(f"NaN values in original streamflow_data: {original_nans}")
if original_nans > 0:
    print("WARNING: Original data contains NaN values!")

# Create wet and dry datasets with new transformation methods

# Extract years from time index and normalize to start from 0
years = np.array([date.year for date in time_index])
years_unique = np.unique(years)
years_normalized = years - years.min()  # start from year 0
n_years = len(years_unique)
n_timesteps = len(time_index)

print("Creating wet and dry non-stationary datasets...")

# WET TRANSFORMATION: Progressive proportional increase based on outflow gauge
# Calculate proportional increase at INK20000 outflow, then apply uniformly to all gauges

# Find the INK20000 outflow gauge column index
streamflow_columns = data_dict["streamflow_columns"]
ink20000_idx = list(streamflow_columns).index("INK20000")

# Initialize the wet transformed data
streamflow_data_wet = streamflow_data.copy()

print(f"Wet target annual addition: {wet_addition_amount:.0f} acre-feet/year")
print(f"Using INK20000 outflow gauge (index {ink20000_idx}) to calculate proportional multipliers")

total_addition = 0
for year in years_unique:
    year_mask = years == year
    year_normalized = year - years.min()
    
    # Calculate progressive addition for this year (0 in first year, full amount in last year)
    if year_normalized == 0:
        annual_addition_target = 0
    else:
        annual_addition_target = wet_addition_amount * (year_normalized / (n_years - 1))
    
    if annual_addition_target > 0:
        # Calculate original annual sum at INK20000 outflow for this year across all runs
        # Shape: (n_runs,) - one value per run
        ink20000_annual_sums = np.sum(streamflow_data[:, year_mask, ink20000_idx], axis=1)
        
        # Calculate proportional multiplier based on INK20000 outflow: (original + addition) / original
        # Shape: (n_runs,) - one multiplier per run
        proportional_multipliers = (ink20000_annual_sums + annual_addition_target) / ink20000_annual_sums
        
        # Apply the same multiplier to ALL gauges uniformly
        # Reshape multiplier to broadcast: (n_runs, 1, 1)
        multipliers_reshaped = proportional_multipliers.reshape(streamflow_data.shape[0], 1, 1)
        
        # Apply to all months and all locations in this year
        streamflow_data_wet[:, year_mask, :] = streamflow_data[:, year_mask, :] * multipliers_reshaped
        total_addition += annual_addition_target
    
    print(f"Year {year}: Annual addition target = {annual_addition_target:.0f} acre-feet")

# DRY TRANSFORMATION: Progressive proportional decrease based on outflow gauge
# Calculate proportional decrease at INK20000 outflow, then apply uniformly to all gauges

# Initialize the dry transformed data
streamflow_data_dry = streamflow_data.copy()

print(f"Dry target annual decrease: {dry_fixed_decrease:.0f} acre-feet/year")
print(f"Dry minimum fraction: {dry_minimum_fraction:.0%} of original value")
print(f"Using INK20000 outflow gauge (index {ink20000_idx}) to calculate proportional multipliers")

for year in years_unique:
    year_mask = years == year
    year_normalized = year - years.min()
    
    # Calculate progressive decrease for this year (0 in first year, full amount in last year)
    if year_normalized == 0:
        annual_decrease_target = 0
    else:
        annual_decrease_target = dry_fixed_decrease * (year_normalized / (n_years - 1))
    
    if annual_decrease_target > 0:
        # Calculate original annual sum at INK20000 outflow for this year across all runs
        # Shape: (n_runs,) - one value per run
        ink20000_annual_sums = np.sum(streamflow_data[:, year_mask, ink20000_idx], axis=1)
        
        # Calculate target annual sum after decrease
        target_annual_sums = ink20000_annual_sums - annual_decrease_target
        
        # Apply minimum fraction protection based on INK20000 outflow
        minimum_annual_sums = ink20000_annual_sums * dry_minimum_fraction
        protected_target_sums = np.maximum(target_annual_sums, minimum_annual_sums)
        
        # Calculate proportional multiplier based on INK20000 outflow: protected_target / original
        # Shape: (n_runs,) - one multiplier per run
        proportional_multipliers = protected_target_sums / ink20000_annual_sums
        
        # Apply the same multiplier to ALL gauges uniformly
        # Reshape multiplier to broadcast: (n_runs, 1, 1)
        multipliers_reshaped = proportional_multipliers.reshape(streamflow_data.shape[0], 1, 1)
        
        # Apply to all months and all locations in this year
        streamflow_data_dry[:, year_mask, :] = streamflow_data[:, year_mask, :] * multipliers_reshaped
    
    print(f"Year {year}: Annual decrease target = {annual_decrease_target:.0f} acre-feet")

print("Wet and dry transformations complete.")
print(f"  Wet: Proportionally increased annual totals from 0 to {wet_addition_amount} acre-feet/year")
print(f"  Dry: Proportionally decreased annual totals from 0 to {dry_fixed_decrease} acre-feet/year (min {dry_minimum_fraction:.0%})")

# Debug: Final check for NaN values in transformed data
print("\n=== DEBUG: Final NaN check after transformations ===")
wet_nans = np.isnan(streamflow_data_wet).sum()
dry_nans = np.isnan(streamflow_data_dry).sum()
print(f"NaN values in wet streamflow data: {wet_nans}")
print(f"NaN values in dry streamflow data: {dry_nans}")

if wet_nans > 0:
    print("ERROR: Wet transformation introduced NaN values!")
if dry_nans > 0:
    print("ERROR: Dry transformation introduced NaN values!")

# Create subset datasets for original, wet, and dry data
print("Creating subset datasets...")

# Original subset dataset
original_subset_dict = {
    "streamflow_data": streamflow_data,  # first 10 runs, original
    "streamflow_index": data_dict["streamflow_index"],
    "streamflow_columns": data_dict["streamflow_columns"]
}
# Note: shortage_data will be generated by WRAP simulation in script 2

# Wet subset dataset (increasing streamflow)
wet_subset_dict = {
    "streamflow_data": streamflow_data_wet,  # first 10 runs, wet trend
    "streamflow_index": data_dict["streamflow_index"],
    "streamflow_columns": data_dict["streamflow_columns"]
}
# Note: shortage_data will be generated by WRAP simulation in script 2

# Dry subset dataset (decreasing streamflow)  
dry_subset_dict = {
    "streamflow_data": streamflow_data_dry,  # first 10 runs, dry trend
    "streamflow_index": data_dict["streamflow_index"],
    "streamflow_columns": data_dict["streamflow_columns"]
}
# Note: shortage_data will be generated by WRAP simulation in script 2

# Save all three datasets
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

dict_to_hdf5(original_subset_path, original_subset_dict)
dict_to_hdf5(wet_subset_path, wet_subset_dict)
dict_to_hdf5(dry_subset_path, dry_subset_dict)

print(f"Original subset dataset saved to: {original_subset_path}")
print(f"Wet subset dataset saved to: {wet_subset_path}")
print(f"Dry subset dataset saved to: {dry_subset_path}")

# Also save the wet dataset with the original name for backward compatibility
dict_to_hdf5(output_dataset_path, wet_subset_dict)

## Create comparison plots ##

# Find the K20000 outflow gauge column
streamflow_columns = data_dict["streamflow_columns"]
k20000_idx = list(streamflow_columns).index("INK20000")

# Extract K20000 streamflow data for all runs
original_k20000 = streamflow_data[:, :, k20000_idx]  # shape: (n_runs, n_timesteps)
wet_k20000 = streamflow_data_wet[:, :, k20000_idx]
dry_k20000 = streamflow_data_dry[:, :, k20000_idx]

# Calculate annual sums for each run and year
n_runs = streamflow_data.shape[0]

# Calculate mean annual sums across all runs (for dashed mean lines)
original_annual_sums_mean = []
wet_annual_sums_mean = []
dry_annual_sums_mean = []

for year in years_unique:
    year_mask = years == year
    # Sum monthly flows for each year for each run, then take mean across runs
    original_year_sums = np.sum(original_k20000[:, year_mask], axis=1)  # sum across months for each run
    wet_year_sums = np.sum(wet_k20000[:, year_mask], axis=1)
    dry_year_sums = np.sum(dry_k20000[:, year_mask], axis=1)
    
    original_annual_sums_mean.append(np.mean(original_year_sums))  # mean across runs
    wet_annual_sums_mean.append(np.mean(wet_year_sums))
    dry_annual_sums_mean.append(np.mean(dry_year_sums))

# Convert to numpy arrays for plotting
original_annual_sums_mean = np.array(original_annual_sums_mean)
wet_annual_sums_mean = np.array(wet_annual_sums_mean)
dry_annual_sums_mean = np.array(dry_annual_sums_mean)

# Note: Removed the mean annual sums plot as requested

# Plot 2: Individual plots for each ensemble member
# Create plots for all 10 ensemble members
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Create separate plots for each ensemble member
for run_idx in range(3):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate annual sums for this specific run
    original_annual_sums_run = []
    wet_annual_sums_run = []
    dry_annual_sums_run = []
    
    for year in years_unique:
        year_mask = years == year
        # Sum monthly flows for each year for this specific run
        original_year_sum = np.sum(original_k20000[run_idx, year_mask])
        wet_year_sum = np.sum(wet_k20000[run_idx, year_mask])
        dry_year_sum = np.sum(dry_k20000[run_idx, year_mask])
        
        original_annual_sums_run.append(original_year_sum)
        wet_annual_sums_run.append(wet_year_sum)
        dry_annual_sums_run.append(dry_year_sum)
    
    # Plot raw annual sums (no log transform)
    original_annual_sums_run = np.array(original_annual_sums_run)
    wet_annual_sums_run = np.array(wet_annual_sums_run)
    dry_annual_sums_run = np.array(dry_annual_sums_run)
    
    # Plot individual run data with solid lines
    ax.plot(years_unique, original_annual_sums_run, 
           color='black', linewidth=2, alpha=0.7, marker='o', markersize=4,
           linestyle='-', label='Original (Run)')
    
    ax.plot(years_unique, wet_annual_sums_run, 
           color='blue', linewidth=2, alpha=0.7, marker='s', markersize=4,
           linestyle='-', label='Wet (Run)')
           
    ax.plot(years_unique, dry_annual_sums_run, 
           color='orange', linewidth=2, alpha=0.7, marker='^', markersize=4,
           linestyle='-', label='Dry (Run)')
    
    # Plot flat mean lines (horizontal dashed lines showing the mean of each time series)
    original_flat_mean = np.mean(original_annual_sums_run)
    wet_flat_mean = np.mean(wet_annual_sums_run)
    dry_flat_mean = np.mean(dry_annual_sums_run)
    
    ax.axhline(y=original_flat_mean, color='black', linewidth=2, alpha=0.8,
               linestyle='--', label='Original (Mean)', zorder=5)
    
    ax.axhline(y=wet_flat_mean, color='blue', linewidth=2, alpha=0.8,
               linestyle='--', label='Wet (Mean)', zorder=5)
           
    ax.axhline(y=dry_flat_mean, color='orange', linewidth=2, alpha=0.8,
               linestyle='--', label='Dry (Mean)', zorder=5)

    ax.set_xlabel('Year', fontsize=16)
    ax.set_ylabel('Annual Streamflow Sum (acre-feet)', fontsize=16)
    # ax.set_title(f'Example Scenario: Annual Sums at Outflow Gauge', fontsize=18)
    ax.legend(fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(join(figure_output_path, f"individual_run_{run_idx+1}_k20000.png"), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Individual ensemble member plots saved to: {figure_output_path}")
print("Files created:")
for run_idx in range(n_runs):
    print(f"  - individual_run_{run_idx+1}_k20000.png")

## Autocorrelation Analysis using Augmented Dickey-Fuller Test ##

def perform_adf_test(data, label):
    """Perform ADF test and return results summary"""

    result = adfuller(data, autolag='AIC')
    adf_stat, p_value, n_lags, n_obs, critical_values, icbest = result
    
    # Determine stationarity based on p-value
    is_stationary = p_value < 0.05
    
    summary = f"""
    {label}:
    ADF Statistic: {adf_stat:.6f}
    p-value: {p_value:.6f}
    Lags used: {n_lags}
    Number of observations: {n_obs}
    Critical Values:
        1%: {critical_values['1%']:.6f}
        5%: {critical_values['5%']:.6f}
        10%: {critical_values['10%']:.6f}
    Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY'} (p < 0.05)
    """
    return summary, is_stationary, p_value


# Test the same 3 sample runs we used for plotting
sample_runs = [0, n_runs//2, n_runs-1]
adf_results = []

for run_idx in range(n_runs):
    print(f"\n--- RUN {run_idx+1} ---")
    
    # Calculate annual sums for this specific run for ADF testing
    original_annual_sums_run = []
    wet_annual_sums_run = []
    dry_annual_sums_run = []
    
    for year in years_unique:
        year_mask = years == year
        # Sum monthly flows for each year for this specific run
        original_year_sum = np.sum(original_k20000[run_idx, year_mask])
        wet_year_sum = np.sum(wet_k20000[run_idx, year_mask])
        dry_year_sum = np.sum(dry_k20000[run_idx, year_mask])
        
        original_annual_sums_run.append(original_year_sum)
        wet_annual_sums_run.append(wet_year_sum)
        dry_annual_sums_run.append(dry_year_sum)
    
    # Convert to numpy arrays
    original_annual_sums_run = np.array(original_annual_sums_run)
    wet_annual_sums_run = np.array(wet_annual_sums_run)
    dry_annual_sums_run = np.array(dry_annual_sums_run)
    
    # Test original annual sums
    orig_summary, orig_stationary, orig_pvalue = perform_adf_test(
        original_annual_sums_run, 
        f"Original Run {run_idx+1} (Annual Sums)"
    )
    print(orig_summary)
    
    # Test wet annual sums
    wet_summary, wet_stationary, wet_pvalue = perform_adf_test(
        wet_annual_sums_run, 
        f"Wet Run {run_idx+1} (Annual Sums)"
    )
    print(wet_summary)
    
    # Test dry annual sums  
    dry_summary, dry_stationary, dry_pvalue = perform_adf_test(
        dry_annual_sums_run, 
        f"Dry Run {run_idx+1} (Annual Sums)"
    )
    print(dry_summary)
    
    # Perform T-tests comparing wet/dry to original
    wet_ttest_stat, wet_ttest_pvalue = ttest_rel(original_annual_sums_run, wet_annual_sums_run)
    dry_ttest_stat, dry_ttest_pvalue = ttest_rel(original_annual_sums_run, dry_annual_sums_run)
    
    print(f"\nT-test results for Run {run_idx+1}:")
    print(f"  Wet vs Original: t={wet_ttest_stat:.4f}, p={wet_ttest_pvalue:.6f}")
    print(f"  Dry vs Original: t={dry_ttest_stat:.4f}, p={dry_ttest_pvalue:.6f}")
    
    # Store results for summary
    adf_results.append({
        'run': run_idx + 1,
        'original_stationary': orig_stationary,
        'original_pvalue': orig_pvalue,
        'wet_stationary': wet_stationary,
        'wet_pvalue': wet_pvalue,
        'dry_stationary': dry_stationary,
        'dry_pvalue': dry_pvalue,
        'wet_ttest_stat': wet_ttest_stat,
        'wet_ttest_pvalue': wet_ttest_pvalue,
        'dry_ttest_stat': dry_ttest_stat,
        'dry_ttest_pvalue': dry_ttest_pvalue,
        'original_mean': np.mean(original_annual_sums_run),
        'wet_mean': np.mean(wet_annual_sums_run),
        'dry_mean': np.mean(dry_annual_sums_run)
    })

# Summary table
print("\n" + "="*80)
print("SUMMARY OF ADF TEST RESULTS")
print("="*80)
print(f"{'Run':<6} {'Original':<12} {'Wet':<12} {'Dry':<12} {'Notes'}")
print("-" * 70)

for result in adf_results:
    if all(result[key] is not None for key in ['original_pvalue', 'wet_pvalue', 'dry_pvalue']):
        orig_status = "Stationary" if result['original_stationary'] else "Non-stat"
        wet_status = "Stationary" if result['wet_stationary'] else "Non-stat"
        dry_status = "Stationary" if result['dry_stationary'] else "Non-stat"
        
        # Determine notes
        notes = []
        if result['original_stationary'] and not result['wet_stationary']:
            notes.append("Wet→Non-stat")
        if result['original_stationary'] and not result['dry_stationary']:
            notes.append("Dry→Non-stat")
        notes_str = ", ".join(notes) if notes else "No change"
              
        print(f"{result['run']:<6} {orig_status:<12} {wet_status:<12} {dry_status:<12} {notes_str}")
        print(f"{'':>6} p={result['original_pvalue']:.3f}{'':>6} p={result['wet_pvalue']:.3f}{'':>6} p={result['dry_pvalue']:.3f}")
    else:
        print(f"{result['run']:<6} {'Error in ADF test'}")

## Create CSV output with all test results ##
print("\n" + "="*80)
print("CREATING CSV OUTPUT WITH ALL TEST RESULTS")
print("="*80)

# Create comprehensive results DataFrame
results_df = pd.DataFrame(adf_results)

# Add additional summary columns
results_df['wet_mean_change_pct'] = ((results_df['wet_mean'] - results_df['original_mean']) / results_df['original_mean'] * 100)
results_df['dry_mean_change_pct'] = ((results_df['dry_mean'] - results_df['original_mean']) / results_df['original_mean'] * 100)

# Add significance indicators (p < 0.05)
results_df['wet_adf_significant'] = results_df['wet_pvalue'] < 0.05
results_df['dry_adf_significant'] = results_df['dry_pvalue'] < 0.05
results_df['wet_ttest_significant'] = results_df['wet_ttest_pvalue'] < 0.05
results_df['dry_ttest_significant'] = results_df['dry_ttest_pvalue'] < 0.05

# Add renamed ADF p-value columns
results_df['original_adf_pvalue'] = results_df['original_pvalue']
results_df['wet_adf_pvalue'] = results_df['wet_pvalue']
results_df['dry_adf_pvalue'] = results_df['dry_pvalue']

# Reorder columns for better readability - simplified version
column_order = [
    'original_mean', 'wet_mean', 'dry_mean', 
    'dry_ttest_pvalue', 'wet_ttest_pvalue', 
    'original_adf_pvalue', 'dry_adf_pvalue', 'wet_adf_pvalue'
]

results_df_ordered = results_df[column_order]

# Save to CSV
csv_output_path = join(figure_output_path, "statistical_test_results.csv")
results_df_ordered.to_csv(csv_output_path, index=False, float_format='%.6f')

print(f"Statistical test results saved to: {csv_output_path}")
print(f"CSV contains {len(results_df_ordered)} runs with the following columns:")
print("  - original_mean, wet_mean, dry_mean")
print("  - dry_ttest_pvalue, wet_ttest_pvalue")
print("  - original_adf_pvalue, dry_adf_pvalue, wet_adf_pvalue")

# Create summary CSV with counts of significant p-values
print("\nCreating summary CSV with significant p-value counts...")

# Define p-value columns to analyze
pvalue_columns = ['dry_ttest_pvalue', 'wet_ttest_pvalue', 'original_adf_pvalue', 'dry_adf_pvalue', 'wet_adf_pvalue']

# Create summary data
summary_data = []
for col in pvalue_columns:
    significant_count = sum(results_df[col] < 0.05)
    total_count = len(results_df[col])
    percentage = (significant_count / total_count) * 100
    
    summary_data.append({
        'test_type': col,
        'significant_count': significant_count,
        'total_count': total_count,
        'percentage_significant': percentage,
        'description': {
            'dry_ttest_pvalue': 'Dry vs Original T-test (p < 0.05)',
            'wet_ttest_pvalue': 'Wet vs Original T-test (p < 0.05)', 
            'original_adf_pvalue': 'Original ADF Test (p < 0.05 = stationary)',
            'dry_adf_pvalue': 'Dry ADF Test (p < 0.05 = stationary)',
            'wet_adf_pvalue': 'Wet ADF Test (p < 0.05 = stationary)'
        }[col]
    })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Save summary CSV
summary_csv_path = join(figure_output_path, "pvalue_significance_summary.csv")
summary_df.to_csv(summary_csv_path, index=False, float_format='%.2f')

print(f"P-value significance summary saved to: {summary_csv_path}")
print(f"Summary contains counts of significant p-values (< 0.05) for each test type")

# Print summary statistics
print(f"\nSUMMARY STATISTICS:")
print(f"Mean change in wet scenario: {results_df['wet_mean_change_pct'].mean():.2f}%")
print(f"Mean change in dry scenario: {results_df['dry_mean_change_pct'].mean():.2f}%")
print(f"Runs where wet transformation created non-stationarity: {sum(results_df['original_stationary'] & ~results_df['wet_stationary'])}/{len(results_df)}")
print(f"Runs where dry transformation created non-stationarity: {sum(results_df['original_stationary'] & ~results_df['dry_stationary'])}/{len(results_df)}")
print(f"Runs with significant wet T-test (p<0.05): {sum(results_df['wet_ttest_significant'])}/{len(results_df)}")
print(f"Runs with significant dry T-test (p<0.05): {sum(results_df['dry_ttest_significant'])}/{len(results_df)}")

# Print detailed p-value significance summary
print(f"\nP-VALUE SIGNIFICANCE SUMMARY:")
for _, row in summary_df.iterrows():
    print(f"  {row['description']}: {row['significant_count']}/{row['total_count']} ({row['percentage_significant']:.1f}%)")

## Historical Colorado .FLO File Analysis ##

# Load the historical Colorado .FLO file
flo_file_path = join(repo_data_path, "colorado-full", "C3.FLO")

historical_flo_df = flo_to_df(flo_file_path)

# Calculate annual sums for INK20000
historical_flo_df.index = pd.to_datetime(historical_flo_df.index)
historical_ink20000 = historical_flo_df["INK20000"].astype(float)

# Group by year and sum
historical_annual_sums = historical_ink20000.groupby(historical_ink20000.index.year).sum()

# Run ADF test on historical annual sums
hist_summary, hist_stationary, hist_pvalue = perform_adf_test(
    historical_annual_sums.values, 
    "Historical INK20000 (Annual Sums)"
)
print(hist_summary)


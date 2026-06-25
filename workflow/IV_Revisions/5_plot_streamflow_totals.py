from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from toolkit.utils.io import hdf5_to_dict
from toolkit import outputs_data_path


## Settings ##

# Dataset parameters
drought_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # All drought levels to analyze
n_subset = 10  # Number of streamflows to process (matching previous scripts)

## Path Configuration ##

# Use local outputs directory in the same folder as this script
local_outputs_path = join(os.path.dirname(__file__), "outputs")

# Original, wet, and dry subset datasets (from local outputs) - using drought_level 0.0 for wet/dry
drought_level = 0.0  # For wet/dry scenarios
original_dataset_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_original_subset.h5"
)

wet_dataset_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_wet_subset.h5"
)

dry_dataset_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_dry_subset.h5"
)

# Original drought datasets (0.1-0.5) from main outputs
drought_original_paths = {}
for drought_level in drought_levels:
    drought_original_paths[drought_level] = join(
        outputs_data_path,
        "synthetic-test",
        f"synthetic_test_dataset_drought_{str(drought_level)}.h5"
    )

## Main Script ##

print("="*80)
print("STREAMFLOW TOTAL ANALYSIS: ORIGINAL vs WET vs DRY + DROUGHT LEVELS")
print("="*80)

# Check if required datasets exist
if not os.path.exists(original_dataset_path):
    raise FileNotFoundError(f"Original subset dataset not found: {original_dataset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

if not os.path.exists(wet_dataset_path):
    raise FileNotFoundError(f"Wet subset dataset not found: {wet_dataset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

if not os.path.exists(dry_dataset_path):
    raise FileNotFoundError(f"Dry subset dataset not found: {dry_dataset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

# Check drought datasets
missing_drought_datasets = []
for drought_level in drought_levels:
    if not os.path.exists(drought_original_paths[drought_level]):
        missing_drought_datasets.append(f"Drought {drought_level}: {drought_original_paths[drought_level]}")

if missing_drought_datasets:
    print("Warning: Some drought datasets are missing:")
    for missing in missing_drought_datasets:
        print(f"  - {missing}")
    print("Continuing with available datasets...")

print(f"Processing {n_subset} streamflows from each dataset")
print(f"Original subset dataset: {original_dataset_path}")
print(f"Wet subset dataset: {wet_dataset_path}")
print(f"Dry subset dataset: {dry_dataset_path}")
print(f"Drought levels to analyze: {drought_levels}")

## Load Streamflow Data ##
print("\n" + "-"*60)
print("LOADING STREAMFLOW DATA")
print("-"*60)

# Load original streamflow data (subset)
print("Loading original streamflow data (subset)...")
original_data_dict = hdf5_to_dict(original_dataset_path)
original_streamflow = original_data_dict['streamflow_data']  # Shape: (n_runs, n_timesteps, n_gauges)

# Load wet streamflow data
print("Loading wet streamflow data...")
wet_data_dict = hdf5_to_dict(wet_dataset_path)
wet_streamflow = wet_data_dict['streamflow_data']  # Shape: (n_runs, n_timesteps, n_gauges)

# Load dry streamflow data
print("Loading dry streamflow data...")
dry_data_dict = hdf5_to_dict(dry_dataset_path)
dry_streamflow = dry_data_dict['streamflow_data']  # Shape: (n_runs, n_timesteps, n_gauges)

# Load drought datasets
print("Loading drought datasets...")
drought_streamflows = {}
for drought_level in drought_levels:
    if os.path.exists(drought_original_paths[drought_level]):
        print(f"  Loading drought {drought_level}...")
        drought_data_dict = hdf5_to_dict(drought_original_paths[drought_level])
        drought_streamflows[drought_level] = drought_data_dict['streamflow_data']
    else:
        print(f"  Skipping drought {drought_level} (file not found)")

# Convert to numpy if they are torch tensors
if hasattr(original_streamflow, 'numpy'):
    original_streamflow = original_streamflow.numpy()
if hasattr(wet_streamflow, 'numpy'):
    wet_streamflow = wet_streamflow.numpy()
if hasattr(dry_streamflow, 'numpy'):
    dry_streamflow = dry_streamflow.numpy()

for drought_level in drought_streamflows:
    if hasattr(drought_streamflows[drought_level], 'numpy'):
        drought_streamflows[drought_level] = drought_streamflows[drought_level].numpy()

print(f"Original streamflow shape: {original_streamflow.shape}")
print(f"Wet streamflow shape: {wet_streamflow.shape}")
print(f"Dry streamflow shape: {dry_streamflow.shape}")
for drought_level in drought_streamflows:
    print(f"Drought {drought_level} streamflow shape: {drought_streamflows[drought_level].shape}")

## Calculate Total Flow for Each Run ##
print("\n" + "-"*60)
print("CALCULATING TOTAL FLOW FOR EACH RUN")
print("-"*60)

# Calculate total flow for each run (sum across all timesteps and gauges)
# Shape: (n_runs, n_timesteps, n_gauges) -> (n_runs,)
original_total_flow = np.sum(original_streamflow, axis=(1, 2))
wet_total_flow = np.sum(wet_streamflow, axis=(1, 2))
dry_total_flow = np.sum(dry_streamflow, axis=(1, 2))

# Calculate total flow for drought datasets
drought_total_flows = {}
for drought_level in drought_streamflows:
    drought_total_flows[drought_level] = np.sum(drought_streamflows[drought_level], axis=(1, 2))

print(f"Original total flow shape: {original_total_flow.shape}")
print(f"Wet total flow shape: {wet_total_flow.shape}")
print(f"Dry total flow shape: {dry_total_flow.shape}")
for drought_level in drought_total_flows:
    print(f"Drought {drought_level} total flow shape: {drought_total_flows[drought_level].shape}")

# Print summary statistics
print(f"\nTotal Flow Summary Statistics:")
print(f"Original (subset) - Mean: {np.mean(original_total_flow):.2f}, Std: {np.std(original_total_flow):.2f}")
print(f"Original (subset) - Min: {np.min(original_total_flow):.2f}, Max: {np.max(original_total_flow):.2f}")

print(f"Wet - Mean: {np.mean(wet_total_flow):.2f}, Std: {np.std(wet_total_flow):.2f}")
print(f"Wet - Min: {np.min(wet_total_flow):.2f}, Max: {np.max(wet_total_flow):.2f}")

print(f"Dry - Mean: {np.mean(dry_total_flow):.2f}, Std: {np.std(dry_total_flow):.2f}")
print(f"Dry - Min: {np.min(dry_total_flow):.2f}, Max: {np.max(dry_total_flow):.2f}")

for drought_level in drought_total_flows:
    flows = drought_total_flows[drought_level]
    print(f"Drought {drought_level} - Mean: {np.mean(flows):.2f}, Std: {np.std(flows):.2f}")
    print(f"Drought {drought_level} - Min: {np.min(flows):.2f}, Max: {np.max(flows):.2f}")

# Calculate percentage changes
wet_pct_change = ((np.mean(wet_total_flow) - np.mean(original_total_flow)) / np.mean(original_total_flow)) * 100
dry_pct_change = ((np.mean(dry_total_flow) - np.mean(original_total_flow)) / np.mean(original_total_flow)) * 100

print(f"\nPercentage Changes vs Original (subset):")
print(f"Wet vs Original: {wet_pct_change:.2f}%")
print(f"Dry vs Original: {dry_pct_change:.2f}%")

# Calculate percentage changes for drought datasets vs original
if 0.0 in drought_total_flows:
    original_full_mean = np.mean(drought_total_flows[0.0])
    print(f"\nPercentage Changes vs Original (full dataset):")
    for drought_level in sorted(drought_total_flows.keys()):
        if drought_level != 0.0:
            drought_mean = np.mean(drought_total_flows[drought_level])
            pct_change = ((drought_mean - original_full_mean) / original_full_mean) * 100
            print(f"Drought {drought_level} vs Original: {pct_change:.2f}%")

## Create Figure Output Directory ##
figure_output_path = join(local_outputs_path, "figures", "streamflow_totals")
if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)

## Create Boxplots ##
print("\n" + "="*80)
print("CREATING STREAMFLOW TOTAL BOXPLOTS")
print("="*80)

# Prepare data for boxplots
print("Preparing data for visualization...")

# Create DataFrame for boxplot
data_for_boxplot = []
for i, flow in enumerate(original_total_flow):
    data_for_boxplot.append({'Scenario': 'Original (subset)', 'Total_Flow': flow})
for i, flow in enumerate(wet_total_flow):
    data_for_boxplot.append({'Scenario': 'Wet', 'Total_Flow': flow})
for i, flow in enumerate(dry_total_flow):
    data_for_boxplot.append({'Scenario': 'Dry', 'Total_Flow': flow})

# Add drought datasets
for drought_level in sorted(drought_total_flows.keys()):
    for i, flow in enumerate(drought_total_flows[drought_level]):
        data_for_boxplot.append({'Scenario': f'Drought {drought_level}', 'Total_Flow': flow})

df = pd.DataFrame(data_for_boxplot)

# 1. Individual Boxplots for Each Drought Scenario
print("Creating individual boxplots for each drought scenario...")

# Create individual boxplots for each drought level
for drought_level in sorted(drought_total_flows.keys()):
    print(f"  Creating boxplot for drought {drought_level}...")
    
    # Prepare data for this specific drought level
    drought_data = []
    
    # Add original subset data
    for i, flow in enumerate(original_total_flow):
        drought_data.append({'Scenario': 'Original (subset)', 'Total_Flow': flow})
    
    # Add wet and dry data
    for i, flow in enumerate(wet_total_flow):
        drought_data.append({'Scenario': 'Wet', 'Total_Flow': flow})
    for i, flow in enumerate(dry_total_flow):
        drought_data.append({'Scenario': 'Dry', 'Total_Flow': flow})
    
    # Add the specific drought level data
    for i, flow in enumerate(drought_total_flows[drought_level]):
        drought_data.append({'Scenario': f'Drought {drought_level}', 'Total_Flow': flow})
    
    drought_df = pd.DataFrame(drought_data)
    
    # Create boxplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define order and colors
    scenario_order = ['Original (subset)', 'Wet', 'Dry', f'Drought {drought_level}']
    colors = ['grey', 'blue', 'orange', 'red']
    
    sns.boxplot(data=drought_df, x='Scenario', y='Total_Flow', ax=ax, 
                order=scenario_order, palette=colors)
    
    ax.set_title(f'Total Streamflow Comparison: Drought {drought_level} Scenario', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=14)
    ax.set_ylabel('Total Flow (acre-feet)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add median values as text
    for i, scenario in enumerate(scenario_order):
        scenario_values = drought_df[drought_df['Scenario'] == scenario]['Total_Flow']
        if not scenario_values.empty:
            median_val = scenario_values.median()
            mean_val = scenario_values.mean()
            ax.text(i, ax.get_ylim()[1] * 0.95, 
                   f'Median: {median_val:.0f}\nMean: {mean_val:.0f}', 
                   ha='center', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(join(figure_output_path, f"total_streamflow_drought_{drought_level}_boxplot.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

# 2. Combined All Scenarios Boxplot (for reference)
print("Creating combined all scenarios boxplot...")
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Define order for scenarios
scenario_order = ['Original (subset)', 'Wet', 'Dry'] + [f'Drought {d}' for d in sorted(drought_total_flows.keys())]

# Create boxplot with custom colors
colors = ['grey', 'blue', 'orange'] + ['red', 'darkred', 'maroon', 'brown', 'saddlebrown', 'sienna'][:len(drought_total_flows)]
sns.boxplot(data=df, x='Scenario', y='Total_Flow', ax=ax, order=scenario_order, palette=colors)

ax.set_title('Total Streamflow Comparison: All Scenarios (Combined)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=14)
ax.set_ylabel('Total Flow (acre-feet)', fontsize=14)
ax.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(join(figure_output_path, "total_streamflow_all_scenarios_combined_boxplot.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 2. Original vs Wet vs Dry Boxplot (subset comparison)
print("Creating original vs wet vs dry boxplot...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Filter data for subset comparison
subset_df = df[df['Scenario'].isin(['Original (subset)', 'Wet', 'Dry'])]

# Create boxplot with custom colors
sns.boxplot(data=subset_df, x='Scenario', y='Total_Flow', ax=ax, 
            palette=['grey', 'blue', 'orange'], order=['Dry', 'Original (subset)', 'Wet'])

ax.set_title('Total Streamflow Comparison: Original vs Wet vs Dry Scenarios', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=14)
ax.set_ylabel('Total Flow (acre-feet)', fontsize=14)
ax.grid(True, alpha=0.3)

# Add median values as text
boxplot_order = ['Dry', 'Original (subset)', 'Wet']  # Match the boxplot order
for scenario in ['Original (subset)', 'Wet', 'Dry']:
    scenario_values = subset_df[subset_df['Scenario'] == scenario]['Total_Flow']
    if not scenario_values.empty:
        median_val = scenario_values.median()
        mean_val = scenario_values.mean()
        ax.text(boxplot_order.index(scenario), 
               ax.get_ylim()[1] * 0.95, 
               f'Median: {median_val:.0f}\nMean: {mean_val:.0f}', 
               ha='center', va='top', fontsize=10, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(join(figure_output_path, "total_streamflow_subset_comparison_boxplot.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 3. Detailed Comparison with Individual Points (subset only)
print("Creating detailed comparison with individual points (subset)...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create boxplot with individual points for subset
sns.boxplot(data=subset_df, x='Scenario', y='Total_Flow', ax=ax, 
            palette=['grey', 'blue', 'orange'], order=['Dry', 'Original (subset)', 'Wet'])
sns.stripplot(data=subset_df, x='Scenario', y='Total_Flow', ax=ax, 
              color='black', alpha=0.6, size=4, order=['Dry', 'Original (subset)', 'Wet'])

ax.set_title('Total Streamflow Comparison with Individual Runs\nOriginal vs Wet vs Dry Scenarios', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=14)
ax.set_ylabel('Total Flow (acre-feet)', fontsize=14)
ax.grid(True, alpha=0.3)

# Add summary statistics
for i, scenario in enumerate(['Dry', 'Original (subset)', 'Wet']):
    scenario_values = subset_df[subset_df['Scenario'] == scenario]['Total_Flow']
    if not scenario_values.empty:
        median_val = scenario_values.median()
        mean_val = scenario_values.mean()
        std_val = scenario_values.std()
        ax.text(i, ax.get_ylim()[1] * 0.95, 
               f'Median: {median_val:.0f}\nMean: {mean_val:.0f}\nStd: {std_val:.0f}', 
               ha='center', va='top', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(join(figure_output_path, "total_streamflow_detailed_comparison.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 3. Violin Plot for Distribution Shape
print("Creating violin plot for distribution analysis...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

sns.violinplot(data=df, x='Scenario', y='Total_Flow', ax=ax, 
               palette=['grey', 'blue', 'orange'], order=['Dry', 'Original', 'Wet'])

ax.set_title('Total Streamflow Distribution Comparison\nOriginal vs Wet vs Dry Scenarios', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=14)
ax.set_ylabel('Total Flow (acre-feet)', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(join(figure_output_path, "total_streamflow_distribution_comparison.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 4. Run-by-Run Comparison
print("Creating run-by-run comparison...")
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Create line plot showing each run
runs = range(len(original_total_flow))
ax.plot(runs, original_total_flow, 'o-', color='grey', linewidth=2, markersize=6, 
        label='Original', alpha=0.8)
ax.plot(runs, wet_total_flow, 'o-', color='blue', linewidth=2, markersize=6, 
        label='Wet', alpha=0.8)
ax.plot(runs, dry_total_flow, 'o-', color='orange', linewidth=2, markersize=6, 
        label='Dry', alpha=0.8)

ax.set_title('Total Streamflow by Run: Original vs Wet vs Dry Scenarios', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Run Number', fontsize=14)
ax.set_ylabel('Total Flow (acre-feet)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add mean lines
ax.axhline(y=np.mean(original_total_flow), color='grey', linestyle='--', alpha=0.7, 
           label=f'Original Mean: {np.mean(original_total_flow):.0f}')
ax.axhline(y=np.mean(wet_total_flow), color='blue', linestyle='--', alpha=0.7, 
           label=f'Wet Mean: {np.mean(wet_total_flow):.0f}')
ax.axhline(y=np.mean(dry_total_flow), color='orange', linestyle='--', alpha=0.7, 
           label=f'Dry Mean: {np.mean(dry_total_flow):.0f}')

plt.tight_layout()
plt.savefig(join(figure_output_path, "total_streamflow_by_run.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

## Create Summary Statistics Table ##
print("\nSaving summary statistics...")

# Create comprehensive summary
summary_data = []
for scenario, flows in [('Original (subset)', original_total_flow), 
                       ('Wet', wet_total_flow), 
                       ('Dry', dry_total_flow)]:
    summary_data.append({
        'Scenario': scenario,
        'Mean': np.mean(flows),
        'Median': np.median(flows),
        'Std': np.std(flows),
        'Min': np.min(flows),
        'Max': np.max(flows),
        'Pct_Change_vs_Original_Subset': 0.0 if scenario == 'Original (subset)' else 
                                        ((np.mean(flows) - np.mean(original_total_flow)) / np.mean(original_total_flow)) * 100
    })

# Add drought datasets
for drought_level in sorted(drought_total_flows.keys()):
    flows = drought_total_flows[drought_level]
    summary_data.append({
        'Scenario': f'Drought {drought_level}',
        'Mean': np.mean(flows),
        'Median': np.median(flows),
        'Std': np.std(flows),
        'Min': np.min(flows),
        'Max': np.max(flows),
        'Pct_Change_vs_Original_Subset': ((np.mean(flows) - np.mean(original_total_flow)) / np.mean(original_total_flow)) * 100
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(join(figure_output_path, "total_streamflow_summary.csv"), index=False)

# Create detailed run-by-run comparison
detailed_data = []
for i in range(len(original_total_flow)):
    detailed_data.append({
        'Run': i + 1,
        'Original_Total_Flow': original_total_flow[i],
        'Wet_Total_Flow': wet_total_flow[i],
        'Dry_Total_Flow': dry_total_flow[i],
        'Wet_vs_Original_Pct': ((wet_total_flow[i] - original_total_flow[i]) / original_total_flow[i]) * 100,
        'Dry_vs_Original_Pct': ((dry_total_flow[i] - original_total_flow[i]) / original_total_flow[i]) * 100
    })

detailed_df = pd.DataFrame(detailed_data)
detailed_df.to_csv(join(figure_output_path, "total_streamflow_by_run_detailed.csv"), index=False)

## Print Summary Results ##
print("\n" + "="*80)
print("STREAMFLOW TOTAL ANALYSIS SUMMARY")
print("="*80)

print(f"\nDataset Information:")
print(f"  - Number of runs: {len(original_total_flow)}")
print(f"  - Number of timesteps per run: {original_streamflow.shape[1]}")
print(f"  - Number of gauges: {original_streamflow.shape[2]}")
print(f"  - Total flow = sum across all timesteps and all gauges")

print(f"\nTotal Flow Summary (acre-feet):")
for _, row in summary_df.iterrows():
    print(f"  {row['Scenario']:>8}: Mean={row['Mean']:>10.0f}, Median={row['Median']:>10.0f}, Std={row['Std']:>8.0f}")

print(f"\nPercentage Changes vs Original (subset):")
for _, row in summary_df.iterrows():
    if row['Scenario'] != 'Original (subset)':
        print(f"  {row['Scenario']:>20}: {row['Pct_Change_vs_Original_Subset']:>6.2f}%")

print(f"\nGenerated Files:")
print(f"  - Individual drought boxplots: total_streamflow_drought_X_boxplot.png (one for each drought level)")
print(f"  - Combined all scenarios boxplot: {figure_output_path}/total_streamflow_all_scenarios_combined_boxplot.png")
print(f"  - Subset comparison boxplot: {figure_output_path}/total_streamflow_subset_comparison_boxplot.png")
print(f"  - Detailed comparison: {figure_output_path}/total_streamflow_detailed_comparison.png")
print(f"  - Distribution comparison: {figure_output_path}/total_streamflow_distribution_comparison.png")
print(f"  - Run-by-run comparison: {figure_output_path}/total_streamflow_by_run.png")
print(f"  - Summary statistics: {figure_output_path}/total_streamflow_summary.csv")
print(f"  - Detailed run data: {figure_output_path}/total_streamflow_by_run_detailed.csv")

print(f"\nAnalysis complete! All streamflow total comparisons have been saved to: {figure_output_path}")

print(f"\nKey Insights:")
print(f"  - Wet scenario shows {wet_pct_change:.1f}% {'increase' if wet_pct_change > 0 else 'decrease'} in total flow")
print(f"  - Dry scenario shows {dry_pct_change:.1f}% {'increase' if dry_pct_change > 0 else 'decrease'} in total flow")
print(f"  - Drought scenarios show progressive changes in total flow")
print(f"  - Boxplots show the distribution of total flows across all runs")
print(f"  - Run-by-run comparison shows how each individual run responds to the scenarios")
print(f"  - All scenarios comparison shows the full range of drought effects")

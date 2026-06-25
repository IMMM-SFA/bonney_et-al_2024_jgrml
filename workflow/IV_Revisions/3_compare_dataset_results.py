from os.path import join
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, SequentialSampler
from toolkit.emulator.dataset import WrapDataset
from toolkit.utils.io import hdf5_to_dict, load_config, dict_to_hdf5
from toolkit import repo_data_path, outputs_data_path
from toolkit.graphics.model import generate_overall_metrics
from toolkit.data.data import filter_dataset

from toolkit.emulator.trainer import Trainer


## Settings ##

# Load in model
# Select model run and checkpoint
run_name = "run_20240916-123332"  # Replace this with the directory name in runs/ containing the trained model of interest
checkpoint_id = "249"  # This is the checkpoint used in the paper and should not be changed if attempting to reproduce results.

# Dataset parameters
drought_level = 0.0  # Should match the drought level used in previous scripts
n_subset = 10  # Number of streamflows to process (matching previous scripts)

# Plot generation options
generate_extra_plots = True  # Set to False to skip generating additional plots in subfolders (faster execution)

## Path Configuration ##

# Use local outputs directory in the same folder as this script
local_outputs_path = join(os.path.dirname(__file__), "outputs")
if not os.path.exists(local_outputs_path):
    os.makedirs(local_outputs_path)

# Model directory (from main outputs)
model_dir = join(repo_data_path, "ml-models", run_name)
config_file = join(model_dir, "config.yaml")

# Original, wet, and dry subset datasets (from local outputs)
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

# Shortage data paths (from WRAP simulation results)
original_shortage_path = join(
    local_outputs_path,
    f"shortages_drought_{str(drought_level)}_original_subset.h5"
)

wet_shortage_path = join(
    local_outputs_path,
    f"shortages_drought_{str(drought_level)}_wet_subset.h5"
)

dry_shortage_path = join(
    local_outputs_path,
    f"shortages_drought_{str(drought_level)}_dry_subset.h5"
)

# Output paths for predictions (save to local outputs)
original_predictions_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_original_subset_{run_name}_predictions.h5"
)

wet_predictions_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_wet_subset_{run_name}_predictions.h5"
)

dry_predictions_path = join(
    local_outputs_path,
    f"synthetic_test_dataset_drought_{str(drought_level)}_dry_subset_{run_name}_predictions.h5"
)

## Main Script ##

print("="*80)
print("GENERATING ML PREDICTIONS FOR DATASET COMPARISON")
print("="*80)

print(f"Extra plots generation: {'ENABLED' if generate_extra_plots else 'DISABLED'}")
if not generate_extra_plots:
    print("  - Will skip generating additional plots in subfolders for faster execution")
    print("  - Will still generate main boxplot comparisons and error metrics")
else:
    print("  - Will generate all plots including detailed comparisons in subfolders")

# Check if required datasets exist
if not os.path.exists(original_dataset_path):
    raise FileNotFoundError(f"Original subset dataset not found: {original_dataset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

if not os.path.exists(wet_dataset_path):
    raise FileNotFoundError(f"Wet subset dataset not found: {wet_dataset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

if not os.path.exists(dry_dataset_path):
    raise FileNotFoundError(f"Dry subset dataset not found: {dry_dataset_path}\nPlease run 1_generate_nonstationary_streamflow.py first.")

# Check if required shortage datasets exist
if not os.path.exists(original_shortage_path):
    raise FileNotFoundError(f"Original shortage data not found: {original_shortage_path}\nPlease run 2_simulate_nonstationary_shortages.py first.")

if not os.path.exists(wet_shortage_path):
    raise FileNotFoundError(f"Wet shortage data not found: {wet_shortage_path}\nPlease run 2_simulate_nonstationary_shortages.py first.")

if not os.path.exists(dry_shortage_path):
    raise FileNotFoundError(f"Dry shortage data not found: {dry_shortage_path}\nPlease run 2_simulate_nonstationary_shortages.py first.")

if not os.path.exists(config_file):
    raise FileNotFoundError(f"Model config not found: {config_file}\nPlease ensure the model run_name is correct.")

print(f"Using model: {run_name}")
print(f"Checkpoint: {checkpoint_id}")
print(f"Processing {n_subset} streamflows from each dataset")
print(f"Original subset dataset: {original_dataset_path}")
print(f"Wet subset dataset: {wet_dataset_path}")
print(f"Dry subset dataset: {dry_dataset_path}")

# Load a reference dataset that has the correct filtering applied (for filtering our new shortage data)
print("\nLoading reference dataset for filtering...")
reference_dataset_path = join(
    outputs_data_path,
    "synthetic-test", 
    f"synthetic_test_dataset_drought_{str(drought_level)}.h5"
)
if not os.path.exists(reference_dataset_path):
    raise FileNotFoundError(f"Reference dataset not found: {reference_dataset_path}\nThis is needed to apply the same filtering to the new shortage data.")

reference_data_dict = hdf5_to_dict(reference_dataset_path)
print(f"Reference dataset has {reference_data_dict['shortage_data'].shape[2]} water rights (filtered)")

# Load config
print("\nLoading model configuration...")
config = load_config(config_file)

# Load Model
print("Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

trainer = Trainer(device, output_dir=model_dir)
trainer.build(config)
trainer.load_checkpoint(checkpoint_id)

print("Model loaded successfully.")

## Generate predictions for original dataset ##
print("\n" + "-"*60)
print("GENERATING PREDICTIONS FOR ORIGINAL DATASET")
print("-"*60)

# Load original test dataset
print("Loading original dataset...")
original_data_dict = hdf5_to_dict(original_dataset_path)
original_shortage_dict = hdf5_to_dict(original_shortage_path)

# Apply filtering to the shortage data to match the filtered format used in training
print("Applying filtering to original shortage data...")
original_filtered_dict = filter_dataset(original_data_dict, original_shortage_dict, reference_data_dict)

# Combine streamflow and shortage data for WrapDataset
original_combined_dict = original_filtered_dict

original_dataset = WrapDataset(original_combined_dict)

print(f"Original dataset shape: {original_data_dict['streamflow_data'].shape}")

# DEBUG: Check original streamflow statistics
original_streamflow = original_filtered_dict['streamflow_data']
# Convert to numpy if it's a tensor
if hasattr(original_streamflow, 'numpy'):
    original_streamflow_np = original_streamflow.numpy()
else:
    original_streamflow_np = original_streamflow
print(f"Original streamflow - Mean: {np.mean(original_streamflow_np):.2f}, Std: {np.std(original_streamflow_np):.2f}")
print(f"Original streamflow - Min: {np.min(original_streamflow_np):.2f}, Max: {np.max(original_streamflow_np):.2f}")

# Use sequential sampler and no shuffling for consistent ordering
test_sampler = SequentialSampler(original_dataset)
test_data_loader = DataLoader(original_dataset, batch_size=10, shuffle=False, sampler=test_sampler)

print("Running ML predictions on original dataset...")
original_results_tuple = trainer.get_targets_and_predictions(test_data_loader)

# Save original predictions
original_pred_dict = {"shortage_predictions": original_results_tuple[2]}
dict_to_hdf5(original_predictions_path, original_pred_dict)
print(f"Original predictions saved to: {original_predictions_path}")

## Generate predictions for wet dataset ##
print("\n" + "-"*60)
print("GENERATING PREDICTIONS FOR WET DATASET")
print("-"*60)

# Load wet test dataset
print("Loading wet dataset...")
wet_data_dict = hdf5_to_dict(wet_dataset_path)
wet_shortage_dict = hdf5_to_dict(wet_shortage_path)

# Apply filtering to the shortage data to match the filtered format used in training
print("Applying filtering to wet shortage data...")
wet_filtered_dict = filter_dataset(wet_data_dict, wet_shortage_dict, reference_data_dict)

# Combine streamflow and shortage data for WrapDataset
wet_combined_dict = wet_filtered_dict

wet_dataset = WrapDataset(wet_combined_dict)

print(f"Wet dataset shape: {wet_data_dict['streamflow_data'].shape}")

# DEBUG: Check wet streamflow statistics
wet_streamflow = wet_filtered_dict['streamflow_data']
# Convert to numpy if it's a tensor
if hasattr(wet_streamflow, 'numpy'):
    wet_streamflow_np = wet_streamflow.numpy()
else:
    wet_streamflow_np = wet_streamflow
print(f"Wet streamflow - Mean: {np.mean(wet_streamflow_np):.2f}, Std: {np.std(wet_streamflow_np):.2f}")
print(f"Wet streamflow - Min: {np.min(wet_streamflow_np):.2f}, Max: {np.max(wet_streamflow_np):.2f}")

# Use sequential sampler and no shuffling for consistent ordering
test_sampler = SequentialSampler(wet_dataset)
test_data_loader = DataLoader(wet_dataset, batch_size=10, shuffle=False, sampler=test_sampler)

print("Running ML predictions on wet dataset...")
wet_results_tuple = trainer.get_targets_and_predictions(test_data_loader)

# Save wet predictions
wet_pred_dict = {"shortage_predictions": wet_results_tuple[2]}
dict_to_hdf5(wet_predictions_path, wet_pred_dict)
print(f"Wet predictions saved to: {wet_predictions_path}")

## Generate predictions for dry dataset ##
print("\n" + "-"*60)
print("GENERATING PREDICTIONS FOR DRY DATASET")
print("-"*60)

# Load dry test dataset
print("Loading dry dataset...")
dry_data_dict = hdf5_to_dict(dry_dataset_path)
dry_shortage_dict = hdf5_to_dict(dry_shortage_path)

# Apply filtering to the shortage data to match the filtered format used in training
print("Applying filtering to dry shortage data...")
dry_filtered_dict = filter_dataset(dry_data_dict, dry_shortage_dict, reference_data_dict)

# Combine streamflow and shortage data for WrapDataset
dry_combined_dict = dry_filtered_dict

dry_dataset = WrapDataset(dry_combined_dict)

print(f"Dry dataset shape: {dry_data_dict['streamflow_data'].shape}")

# DEBUG: Check dry streamflow statistics
dry_streamflow = dry_filtered_dict['streamflow_data']
# Convert to numpy if it's a tensor
if hasattr(dry_streamflow, 'numpy'):
    dry_streamflow_np = dry_streamflow.numpy()
else:
    dry_streamflow_np = dry_streamflow
print(f"Dry streamflow - Mean: {np.mean(dry_streamflow_np):.2f}, Std: {np.std(dry_streamflow_np):.2f}")
print(f"Dry streamflow - Min: {np.min(dry_streamflow_np):.2f}, Max: {np.max(dry_streamflow_np):.2f}")

# Use sequential sampler and no shuffling for consistent ordering
test_sampler = SequentialSampler(dry_dataset)
test_data_loader = DataLoader(dry_dataset, batch_size=10, shuffle=False, sampler=test_sampler)

print("Running ML predictions on dry dataset...")
dry_results_tuple = trainer.get_targets_and_predictions(test_data_loader)

# Save dry predictions
dry_pred_dict = {"shortage_predictions": dry_results_tuple[2]}
dict_to_hdf5(dry_predictions_path, dry_pred_dict)
print(f"Dry predictions saved to: {dry_predictions_path}")

## CRITICAL DEBUG: Compare streamflow inputs to ML model ##
print("\n" + "="*80)
print("DEBUGGING: CHECKING IF STREAMFLOW INPUTS ARE ACTUALLY DIFFERENT")
print("="*80)

# Extract streamflow data that was actually fed to the ML model
original_ml_streamflow = original_results_tuple[0]  # inputs from get_targets_and_predictions
wet_ml_streamflow = wet_results_tuple[0]           # inputs from get_targets_and_predictions  
dry_ml_streamflow = dry_results_tuple[0]           # inputs from get_targets_and_predictions

print(f"Original ML input streamflow shape: {original_ml_streamflow.shape}")
print(f"Wet ML input streamflow shape: {wet_ml_streamflow.shape}")
print(f"Dry ML input streamflow shape: {dry_ml_streamflow.shape}")

# Check if streamflow inputs are identical
original_vs_wet = np.allclose(original_ml_streamflow, wet_ml_streamflow, atol=1e-6)
original_vs_dry = np.allclose(original_ml_streamflow, dry_ml_streamflow, atol=1e-6)
wet_vs_dry = np.allclose(wet_ml_streamflow, dry_ml_streamflow, atol=1e-6)

print(f"\nStreamflow comparison results:")
print(f"Original vs Wet IDENTICAL: {original_vs_wet}")
print(f"Original vs Dry IDENTICAL: {original_vs_dry}")
print(f"Wet vs Dry IDENTICAL: {wet_vs_dry}")

if original_vs_wet and original_vs_dry:
    print("\n🚨 CRITICAL ISSUE: All streamflow inputs to ML model are IDENTICAL!")
    print("   This explains why predictions are the same across datasets.")
else:
    print("\n✅ Streamflow inputs are different - investigating predictions...")

# Also check if predictions are identical
original_predictions = original_results_tuple[2]
wet_predictions = wet_results_tuple[2]
dry_predictions = dry_results_tuple[2]

pred_original_vs_wet = np.allclose(original_predictions, wet_predictions, atol=1e-6)
pred_original_vs_dry = np.allclose(original_predictions, dry_predictions, atol=1e-6)
pred_wet_vs_dry = np.allclose(wet_predictions, dry_predictions, atol=1e-6)

print(f"\nPrediction comparison results:")
print(f"Original vs Wet predictions IDENTICAL: {pred_original_vs_wet}")
print(f"Original vs Dry predictions IDENTICAL: {pred_original_vs_dry}")
print(f"Wet vs Dry predictions IDENTICAL: {pred_wet_vs_dry}")

print("="*80)

## Calculate Error Metrics and Create Boxplots ##
print("\n" + "="*80)
print("CALCULATING ERROR METRICS AND CREATING COMPARISON BOXPLOTS")
print("="*80)

# Create figure output directory
figure_output_path = join(local_outputs_path, "figures", "prediction_comparison")
if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)

# Get water right labels from the filtered dataset
print("Loading water right labels...")
# Use shortage columns from the filtered dataset (this should match the model training data)
if 'shortage_columns' in original_combined_dict:
    right_labels = original_combined_dict['shortage_columns']
else:
    # Fallback if not available
    print("Warning: Using simplified approach for right labels")
    right_labels = [f"right_{i}" for i in range(original_results_tuple[1].shape[2])]

print(f"Using {len(right_labels)} water rights for error calculation")

# Calculate error metrics for each dataset
print("\nCalculating error metrics...")

print("  - Original dataset metrics...")
original_metrics, original_vol_metrics = generate_overall_metrics(original_results_tuple, right_labels)

print("  - Wet dataset metrics...")
wet_metrics, wet_vol_metrics = generate_overall_metrics(wet_results_tuple, right_labels)

print("  - Dry dataset metrics...")
dry_metrics, dry_vol_metrics = generate_overall_metrics(dry_results_tuple, right_labels)

# Prepare data for boxplots
error_metrics_list = ['overall_mse', 'overall_mae', 'overall_me', 'overall_nse']
volumetric_metrics_list = ['overall_volumetric_mse', 'overall_volumetric_mae', 'overall_volumetric_me', 'overall_volumetric_nse']

# Create dataframes for plotting
print("\nPreparing data for visualization...")

# Standard error metrics
error_data = []
for metric in error_metrics_list:
    # Add original data
    for value in original_metrics[metric]:
        if not np.isnan(value):
            error_data.append({'Dataset': 'Original', 'Metric': metric, 'Value': value})
    
    # Add wet data
    for value in wet_metrics[metric]:
        if not np.isnan(value):
            error_data.append({'Dataset': 'Wet', 'Metric': metric, 'Value': value})
    
    # Add dry data
    for value in dry_metrics[metric]:
        if not np.isnan(value):
            error_data.append({'Dataset': 'Dry', 'Metric': metric, 'Value': value})

error_df = pd.DataFrame(error_data)

# Volumetric error metrics
vol_error_data = []
for metric in volumetric_metrics_list:
    # Add original data
    for value in original_vol_metrics[metric]:
        if not np.isnan(value):
            vol_error_data.append({'Dataset': 'Original', 'Metric': metric, 'Value': value})
    
    # Add wet data
    for value in wet_vol_metrics[metric]:
        if not np.isnan(value):
            vol_error_data.append({'Dataset': 'Wet', 'Metric': metric, 'Value': value})
    
    # Add dry data
    for value in dry_vol_metrics[metric]:
        if not np.isnan(value):
            vol_error_data.append({'Dataset': 'Dry', 'Metric': metric, 'Value': value})

vol_error_df = pd.DataFrame(vol_error_data)

# Create boxplots
print("\nCreating boxplot visualizations...")

# 1. Standard Error Metrics Boxplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('ML Model Error Metrics Comparison\nOriginal vs. Wet vs. Dry Scenarios', fontsize=16, fontweight='bold')

metrics_info = [
    ('overall_mse', 'Mean Squared Error (MSE)', 'MSE'),
    ('overall_mae', 'Mean Absolute Error (MAE)', 'MAE'), 
    ('overall_me', 'Mean Error (ME)', 'ME'),
    ('overall_nse', 'Nash-Sutcliffe Efficiency (NSE)', 'NSE')
]

for i, (metric, title, ylabel) in enumerate(metrics_info):
    ax = axes[i//2, i%2]
    metric_data = error_df[error_df['Metric'] == metric]
    
    if not metric_data.empty:
        sns.boxplot(data=metric_data, x='Dataset', y='Value', ax=ax, palette=['orange', 'grey', 'blue'], order=['Dry', 'Original', 'Wet'])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add median values as text
        boxplot_order = ['Dry', 'Original', 'Wet']  # Match the boxplot order
        for dataset in ['Original', 'Wet', 'Dry']:
            dataset_values = metric_data[metric_data['Dataset'] == dataset]['Value']
            if not dataset_values.empty:
                median_val = dataset_values.median()
                ax.text(boxplot_order.index(dataset), 
                       ax.get_ylim()[1] * 0.95, 
                       f'Median: {median_val:.4f}', 
                       ha='center', va='top', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(join(figure_output_path, "error_metrics_comparison_boxplots.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Volumetric Error Metrics Boxplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ML Model Volumetric Error Metrics Comparison\nOriginal vs. Wet vs. Dry Scenarios', fontsize=16, fontweight='bold')

vol_metrics_info = [
    ('overall_volumetric_mse', 'Volumetric Mean Squared Error', 'Volumetric MSE'),
    ('overall_volumetric_mae', 'Volumetric Mean Absolute Error', 'Volumetric MAE'),
    ('overall_volumetric_me', 'Volumetric Mean Error', 'Volumetric ME'),
    ('overall_volumetric_nse', 'Volumetric Nash-Sutcliffe Efficiency', 'Volumetric NSE')
]

for i, (metric, title, ylabel) in enumerate(vol_metrics_info):
    ax = axes[i//2, i%2]
    metric_data = vol_error_df[vol_error_df['Metric'] == metric]
    
    if not metric_data.empty:
        sns.boxplot(data=metric_data, x='Dataset', y='Value', ax=ax, palette=['orange', 'grey', 'blue'], order=['Dry', 'Original', 'Wet'])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add median values as text
        boxplot_order = ['Dry', 'Original', 'Wet']  # Match the boxplot order
        for dataset in ['Original', 'Wet', 'Dry']:
            dataset_values = metric_data[metric_data['Dataset'] == dataset]['Value']
            if not dataset_values.empty:
                median_val = dataset_values.median()
                ax.text(boxplot_order.index(dataset), 
                       ax.get_ylim()[1] * 0.95, 
                       f'Median: {median_val:.4f}', 
                       ha='center', va='top', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(join(figure_output_path, "volumetric_error_metrics_comparison_boxplots.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Combined comparison plot for key metrics
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Key ML Model Performance Metrics\nOriginal vs. Wet vs. Dry Scenarios', fontsize=16, fontweight='bold')

# MAE comparison
mae_data = error_df[error_df['Metric'] == 'overall_mae']
sns.boxplot(data=mae_data, x='Dataset', y='Value', ax=axes[0], palette=['orange', 'grey', 'blue'], order=['Dry', 'Original', 'Wet'])
axes[0].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('MAE', fontsize=12)
axes[0].set_xlabel('Dataset', fontsize=12)
axes[0].grid(True, alpha=0.3)

# NSE comparison
nse_data = error_df[error_df['Metric'] == 'overall_nse']
sns.boxplot(data=nse_data, x='Dataset', y='Value', ax=axes[1], palette=['orange', 'grey', 'blue'], order=['Dry', 'Original', 'Wet'])
axes[1].set_title('Nash-Sutcliffe Efficiency (NSE)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('NSE', fontsize=12)
axes[1].set_xlabel('Dataset', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(join(figure_output_path, "key_metrics_comparison_boxplots.png"), dpi=300, bbox_inches='tight')
plt.close()

# Save error metrics summary to CSV
print("\nSaving error metrics summary...")
summary_stats = []

for dataset_name, metrics, vol_metrics in [('Original', original_metrics, original_vol_metrics),
                                          ('Wet', wet_metrics, wet_vol_metrics), 
                                          ('Dry', dry_metrics, dry_vol_metrics)]:
    for metric_name, values in metrics.items():
        summary_stats.append({
            'Dataset': dataset_name,
            'Metric': metric_name,
            'Mean': np.nanmean(values),
            'Median': np.nanmedian(values),
            'Std': np.nanstd(values),
            'Min': np.nanmin(values),
            'Max': np.nanmax(values)
        })
    
    for metric_name, values in vol_metrics.items():
        summary_stats.append({
            'Dataset': dataset_name,
            'Metric': metric_name,
            'Mean': np.nanmean(values),
            'Median': np.nanmedian(values),
            'Std': np.nanstd(values),
            'Min': np.nanmin(values),
            'Max': np.nanmax(values)
        })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(join(figure_output_path, "error_metrics_summary.csv"), index=False)

print(f"Error metrics boxplots saved to: {figure_output_path}")
print("Generated plots:")
print("  1. error_metrics_comparison_boxplots.png - Standard error metrics")
print("  2. volumetric_error_metrics_comparison_boxplots.png - Volumetric error metrics")
print("  3. key_metrics_comparison_boxplots.png - Key metrics (MAE & NSE)")
print("  4. error_metrics_summary.csv - Statistical summary")

## Create imshow comparison plots for wet dataset ##
if generate_extra_plots:
    print("\n" + "="*80)
    print("CREATING IMSHOW COMPARISONS FOR WET DATASET")
    print("="*80)

    # Create wet comparison output directory
    wet_comparison_path = join(figure_output_path, "wet_comparisons")
    if not os.path.exists(wet_comparison_path):
        os.makedirs(wet_comparison_path)

    # Extract wet dataset expected and predicted results
    wet_expected = wet_results_tuple[1]  # True values (shape: n_runs, n_timesteps, n_rights)
    wet_predicted = wet_results_tuple[2]  # Predicted values (shape: n_runs, n_timesteps, n_rights)

    print(f"Wet expected shape: {wet_expected.shape}")
    print(f"Wet predicted shape: {wet_predicted.shape}")

    # Create comparison plots for each run
    n_runs_wet = wet_expected.shape[0]
    print(f"Creating {n_runs_wet} comparison plots for wet dataset...")

    for run_idx in range(3):
        print(f"  Creating plot for run {run_idx + 1}...")
        
        # Extract data for this run
        expected_run = wet_expected[run_idx]  # Shape: (n_timesteps, n_rights)
        predicted_run = wet_predicted[run_idx]  # Shape: (n_timesteps, n_rights)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Wet Dataset - Run {run_idx + 1}: Expected vs Predicted Water Shortages', 
                     fontsize=16, fontweight='bold')
        
        # Find common color scale for consistency
        vmin = min(expected_run.min(), predicted_run.min())
        vmax = max(expected_run.max(), predicted_run.max())
        
        # Plot 1: Expected (True) values
        im1 = axes[0].imshow(expected_run.T, aspect='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Expected (True) Shortages', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Steps', fontsize=12)
        axes[0].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im1, ax=axes[0], label='Shortage Ratio')
        
        # Plot 2: Predicted values
        im2 = axes[1].imshow(predicted_run.T, aspect='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title('Predicted Shortages', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Steps', fontsize=12)
        axes[1].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im2, ax=axes[1], label='Shortage Ratio')
        
        # Plot 3: Difference (Predicted - Expected)
        difference = predicted_run - expected_run
        diff_max = max(abs(difference.min()), abs(difference.max()))
        im3 = axes[2].imshow(difference.T, aspect='auto', cmap='RdBu_r', 
                            vmin=-diff_max, vmax=diff_max, origin='lower')
        axes[2].set_title('Difference (Predicted - Expected)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time Steps', fontsize=12)
        axes[2].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im3, ax=axes[2], label='Shortage Difference')
        
        # Add summary statistics as text
        mae_run = np.mean(np.abs(difference))
        rmse_run = np.sqrt(np.mean(difference**2))
        
        fig.text(0.02, 0.02, f'MAE: {mae_run:.4f} | RMSE: {rmse_run:.4f}', 
                 fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(join(wet_comparison_path, f"wet_run_{run_idx + 1}_comparison.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create summary comparison plot (average across all runs)
    print("Creating summary comparison plot (averaged across all runs)...")

    # Calculate averages across all runs
    expected_avg = np.mean(wet_expected, axis=0)  # Shape: (n_timesteps, n_rights)
    predicted_avg = np.mean(wet_predicted, axis=0)  # Shape: (n_timesteps, n_rights)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Wet Dataset - Average Across All Runs: Expected vs Predicted Water Shortages', 
                 fontsize=16, fontweight='bold')

    # Find common color scale
    vmin = min(expected_avg.min(), predicted_avg.min())
    vmax = max(expected_avg.max(), predicted_avg.max())

    # Plot 1: Average Expected values
    im1 = axes[0].imshow(expected_avg.T, aspect='auto', cmap='viridis', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Average Expected Shortages', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Shortage Ratio')

    # Plot 2: Average Predicted values
    im2 = axes[1].imshow(predicted_avg.T, aspect='auto', cmap='viridis', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Average Predicted Shortages', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Steps', fontsize=12)
    axes[1].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Shortage Ratio')

    # Plot 3: Average Difference
    difference_avg = predicted_avg - expected_avg
    diff_max = max(abs(difference_avg.min()), abs(difference_avg.max()))
    im3 = axes[2].imshow(difference_avg.T, aspect='auto', cmap='RdBu_r', 
                        vmin=-diff_max, vmax=diff_max, origin='lower')
    axes[2].set_title('Average Difference (Predicted - Expected)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Steps', fontsize=12)
    axes[2].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im3, ax=axes[2], label='Shortage Difference')

    # Add summary statistics
    mae_avg = np.mean(np.abs(difference_avg))
    rmse_avg = np.sqrt(np.mean(difference_avg**2))

    fig.text(0.02, 0.02, f'Average MAE: {mae_avg:.4f} | Average RMSE: {rmse_avg:.4f}', 
             fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(join(wet_comparison_path, "wet_average_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Wet dataset comparison plots saved to: {wet_comparison_path}")
    print(f"Generated {n_runs_wet + 1} comparison plots:")
    print(f"  - Individual run comparisons: wet_run_1_comparison.png to wet_run_{n_runs_wet}_comparison.png")
    print(f"  - Average comparison: wet_average_comparison.png")
else:
    print("\nSkipping wet dataset imshow comparison plots (generate_extra_plots = False)")

## Create imshow comparison plots for original dataset ##
if generate_extra_plots:
    print("\n" + "="*80)
    print("CREATING IMSHOW COMPARISONS FOR ORIGINAL DATASET")
    print("="*80)

    # Create original comparison output directory
    original_comparison_path = join(figure_output_path, "original_comparisons")
    if not os.path.exists(original_comparison_path):
        os.makedirs(original_comparison_path)

    # Extract original dataset expected and predicted results
    original_expected = original_results_tuple[1]  # True values
    original_predicted = original_results_tuple[2]  # Predicted values

    print(f"Original expected shape: {original_expected.shape}")
    print(f"Original predicted shape: {original_predicted.shape}")

    # Create comparison plots for each run
    n_runs_original = original_expected.shape[0]
    print(f"Creating {n_runs_original} comparison plots for original dataset...")

    for run_idx in range(3):
        print(f"  Creating plot for run {run_idx + 1}...")
        
        # Extract data for this run
        expected_run = original_expected[run_idx]
        predicted_run = original_predicted[run_idx]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Original Dataset - Run {run_idx + 1}: Expected vs Predicted Water Shortages', 
                     fontsize=16, fontweight='bold')
    
        # Find common color scale for consistency
        vmin = min(expected_run.min(), predicted_run.min())
        vmax = max(expected_run.max(), predicted_run.max())
        
        # Plot 1: Expected (True) values
        im1 = axes[0].imshow(expected_run.T, aspect='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Expected (True) Shortages', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Steps', fontsize=12)
        axes[0].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im1, ax=axes[0], label='Shortage Ratio')
        
        # Plot 2: Predicted values
        im2 = axes[1].imshow(predicted_run.T, aspect='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title('Predicted Shortages', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Steps', fontsize=12)
        axes[1].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im2, ax=axes[1], label='Shortage Ratio')
        
        # Plot 3: Difference (Predicted - Expected)
        difference = predicted_run - expected_run
        diff_max = max(abs(difference.min()), abs(difference.max()))
        im3 = axes[2].imshow(difference.T, aspect='auto', cmap='RdBu_r', 
                            vmin=-diff_max, vmax=diff_max, origin='lower')
        axes[2].set_title('Difference (Predicted - Expected)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time Steps', fontsize=12)
        axes[2].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im3, ax=axes[2], label='Shortage Difference')
        
        # Add summary statistics as text
        mae_run = np.mean(np.abs(difference))
        rmse_run = np.sqrt(np.mean(difference**2))
        
        fig.text(0.02, 0.02, f'MAE: {mae_run:.4f} | RMSE: {rmse_run:.4f}', 
                fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(join(original_comparison_path, f"original_run_{run_idx + 1}_comparison.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create summary comparison plot for original (average across all runs)
    print("Creating summary comparison plot for original dataset (averaged across all runs)...")

    # Calculate averages across all runs
    expected_avg = np.mean(original_expected, axis=0)
    predicted_avg = np.mean(original_predicted, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Original Dataset - Average Across All Runs: Expected vs Predicted Water Shortages', 
                fontsize=16, fontweight='bold')

    # Find common color scale
    vmin = min(expected_avg.min(), predicted_avg.min())
    vmax = max(expected_avg.max(), predicted_avg.max())

    # Plot 1: Average Expected values
    im1 = axes[0].imshow(expected_avg.T, aspect='auto', cmap='viridis', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Average Expected Shortages', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Shortage Ratio')

    # Plot 2: Average Predicted values
    im2 = axes[1].imshow(predicted_avg.T, aspect='auto', cmap='viridis', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Average Predicted Shortages', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Steps', fontsize=12)
    axes[1].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Shortage Ratio')

    # Plot 3: Average Difference
    difference_avg = predicted_avg - expected_avg
    diff_max = max(abs(difference_avg.min()), abs(difference_avg.max()))
    im3 = axes[2].imshow(difference_avg.T, aspect='auto', cmap='RdBu_r', 
                        vmin=-diff_max, vmax=diff_max, origin='lower')
    axes[2].set_title('Average Difference (Predicted - Expected)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Steps', fontsize=12)
    axes[2].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im3, ax=axes[2], label='Shortage Difference')

    # Add summary statistics
    mae_avg = np.mean(np.abs(difference_avg))
    rmse_avg = np.sqrt(np.mean(difference_avg**2))

    fig.text(0.02, 0.02, f'Average MAE: {mae_avg:.4f} | Average RMSE: {rmse_avg:.4f}', 
            fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(join(original_comparison_path, "original_average_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Original dataset comparison plots saved to: {original_comparison_path}")
    print(f"Generated {n_runs_original + 1} comparison plots:")
    print(f"  - Individual run comparisons: original_run_1_comparison.png to original_run_{n_runs_original}_comparison.png")
    print(f"  - Average comparison: original_average_comparison.png")
else:
    print("\nSkipping original dataset imshow comparison plots (generate_extra_plots = False)")

## Create imshow comparison plots for dry dataset ##
if generate_extra_plots:
    print("\n" + "="*80)
    print("CREATING IMSHOW COMPARISONS FOR DRY DATASET")
    print("="*80)

    # Create dry comparison output directory
    dry_comparison_path = join(figure_output_path, "dry_comparisons")
    if not os.path.exists(dry_comparison_path):
        os.makedirs(dry_comparison_path)

    # Extract dry dataset expected and predicted results
    dry_expected = dry_results_tuple[1]  # True values
    dry_predicted = dry_results_tuple[2]  # Predicted values

    print(f"Dry expected shape: {dry_expected.shape}")
    print(f"Dry predicted shape: {dry_predicted.shape}")

    # Create comparison plots for each run
    n_runs_dry = dry_expected.shape[0]
    print(f"Creating {n_runs_dry} comparison plots for dry dataset...")

    for run_idx in range(3):
        print(f"  Creating plot for run {run_idx + 1}...")
        
        # Extract data for this run
        expected_run = dry_expected[run_idx]
        predicted_run = dry_predicted[run_idx]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Dry Dataset - Run {run_idx + 1}: Expected vs Predicted Water Shortages', 
                    fontsize=16, fontweight='bold')
        
        # Find common color scale for consistency
        vmin = min(expected_run.min(), predicted_run.min())
        vmax = max(expected_run.max(), predicted_run.max())
        
        # Plot 1: Expected (True) values
        im1 = axes[0].imshow(expected_run.T, aspect='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Expected (True) Shortages', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Steps', fontsize=12)
        axes[0].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im1, ax=axes[0], label='Shortage Ratio')
        
        # Plot 2: Predicted values
        im2 = axes[1].imshow(predicted_run.T, aspect='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title('Predicted Shortages', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Steps', fontsize=12)
        axes[1].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im2, ax=axes[1], label='Shortage Ratio')
        
        # Plot 3: Difference (Predicted - Expected)
        difference = predicted_run - expected_run
        diff_max = max(abs(difference.min()), abs(difference.max()))
        im3 = axes[2].imshow(difference.T, aspect='auto', cmap='RdBu_r', 
                            vmin=-diff_max, vmax=diff_max, origin='lower')
        axes[2].set_title('Difference (Predicted - Expected)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time Steps', fontsize=12)
        axes[2].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im3, ax=axes[2], label='Shortage Difference')
        
        # Add summary statistics as text
        mae_run = np.mean(np.abs(difference))
        rmse_run = np.sqrt(np.mean(difference**2))
        
        fig.text(0.02, 0.02, f'MAE: {mae_run:.4f} | RMSE: {rmse_run:.4f}', 
                fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(join(dry_comparison_path, f"dry_run_{run_idx + 1}_comparison.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create summary comparison plot for dry (average across all runs)
    print("Creating summary comparison plot for dry dataset (averaged across all runs)...")

    # Calculate averages across all runs
    expected_avg = np.mean(dry_expected, axis=0)
    predicted_avg = np.mean(dry_predicted, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Dry Dataset - Average Across All Runs: Expected vs Predicted Water Shortages', 
                fontsize=16, fontweight='bold')

    # Find common color scale
    vmin = min(expected_avg.min(), predicted_avg.min())
    vmax = max(expected_avg.max(), predicted_avg.max())

    # Plot 1: Average Expected values
    im1 = axes[0].imshow(expected_avg.T, aspect='auto', cmap='viridis', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Average Expected Shortages', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Shortage Ratio')

    # Plot 2: Average Predicted values
    im2 = axes[1].imshow(predicted_avg.T, aspect='auto', cmap='viridis', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Average Predicted Shortages', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Steps', fontsize=12)
    axes[1].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Shortage Ratio')

    # Plot 3: Average Difference
    difference_avg = predicted_avg - expected_avg
    diff_max = max(abs(difference_avg.min()), abs(difference_avg.max()))
    im3 = axes[2].imshow(difference_avg.T, aspect='auto', cmap='RdBu_r', 
                        vmin=-diff_max, vmax=diff_max, origin='lower')
    axes[2].set_title('Average Difference (Predicted - Expected)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Steps', fontsize=12)
    axes[2].set_ylabel('Water Rights', fontsize=12)
    plt.colorbar(im3, ax=axes[2], label='Shortage Difference')

    # Add summary statistics
    mae_avg = np.mean(np.abs(difference_avg))
    rmse_avg = np.sqrt(np.mean(difference_avg**2))

    fig.text(0.02, 0.02, f'Average MAE: {mae_avg:.4f} | Average RMSE: {rmse_avg:.4f}', 
            fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(join(dry_comparison_path, "dry_average_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Dry dataset comparison plots saved to: {dry_comparison_path}")
    print(f"Generated {n_runs_dry + 1} comparison plots:")
    print(f"  - Individual run comparisons: dry_run_1_comparison.png to dry_run_{n_runs_dry}_comparison.png")
    print(f"  - Average comparison: dry_average_comparison.png")
else:
    print("\nSkipping dry dataset imshow comparison plots (generate_extra_plots = False)")

## Create streamflow comparison plots for each run ##
if generate_extra_plots:
    print("\n" + "="*80)
    print("CREATING STREAMFLOW COMPARISONS FOR EACH RUN")
    print("="*80)

# Create streamflow comparison output directory
streamflow_comparison_path = join(figure_output_path, "streamflow_comparisons")
if not os.path.exists(streamflow_comparison_path):
    os.makedirs(streamflow_comparison_path)

# Extract streamflow data from all three datasets and convert to numpy
original_streamflow = original_combined_dict['streamflow_data']  # Shape: (n_runs, n_timesteps, n_gauges)
wet_streamflow = wet_combined_dict['streamflow_data']  # Shape: (n_runs, n_timesteps, n_gauges)
dry_streamflow = dry_combined_dict['streamflow_data']  # Shape: (n_runs, n_timesteps, n_gauges)

# Convert to numpy if they are torch tensors
if hasattr(original_streamflow, 'numpy'):
    original_streamflow = original_streamflow.numpy()
if hasattr(wet_streamflow, 'numpy'):
    wet_streamflow = wet_streamflow.numpy()
if hasattr(dry_streamflow, 'numpy'):
    dry_streamflow = dry_streamflow.numpy()

print(f"Original streamflow shape: {original_streamflow.shape}")
print(f"Wet streamflow shape: {wet_streamflow.shape}")
print(f"Dry streamflow shape: {dry_streamflow.shape}")

n_runs_streamflow = original_streamflow.shape[0]
print(f"Creating {n_runs_streamflow} streamflow comparison plots...")

for run_idx in range(3):
    print(f"  Creating streamflow plot for run {run_idx + 1}...")
    
    # Extract streamflow data for this run
    original_run = original_streamflow[run_idx]  # Shape: (n_timesteps, n_gauges)
    wet_run = wet_streamflow[run_idx]  # Shape: (n_timesteps, n_gauges)
    dry_run = dry_streamflow[run_idx]  # Shape: (n_timesteps, n_gauges)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Run {run_idx + 1}: Streamflow Comparison (Original vs Wet vs Dry)', 
                 fontsize=16, fontweight='bold')
    
    # Find common color scale for consistency
    vmin = min(original_run.min(), wet_run.min(), dry_run.min())
    vmax = max(original_run.max(), wet_run.max(), dry_run.max())
    
    # Plot 1: Original streamflow
    im1 = axes[0].imshow(original_run.T, aspect='auto', cmap='Blues', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Original Streamflow', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Gauge Locations', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Streamflow (acre-feet)')
    
    # Plot 2: Wet streamflow
    im2 = axes[1].imshow(wet_run.T, aspect='auto', cmap='Blues', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Wet Streamflow (Increasing)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Steps', fontsize=12)
    axes[1].set_ylabel('Gauge Locations', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Streamflow (acre-feet)')
    
    # Plot 3: Dry streamflow
    im3 = axes[2].imshow(dry_run.T, aspect='auto', cmap='Blues', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[2].set_title('Dry Streamflow (Decreasing)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Steps', fontsize=12)
    axes[2].set_ylabel('Gauge Locations', fontsize=12)
    plt.colorbar(im3, ax=axes[2], label='Streamflow (acre-feet)')
    
    # Add summary statistics
    original_mean = np.mean(original_run)
    wet_mean = np.mean(wet_run)
    dry_mean = np.mean(dry_run)
    
    fig.text(0.02, 0.02, f'Mean Streamflow - Original: {original_mean:.2f} | Wet: {wet_mean:.2f} | Dry: {dry_mean:.2f}', 
             fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(join(streamflow_comparison_path, f"streamflow_run_{run_idx + 1}_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Streamflow comparison plots saved to: {streamflow_comparison_path}")
else:
    print("\nSkipping streamflow comparison plots (generate_extra_plots = False)")

## Create shortage comparison plots for each run ##
if generate_extra_plots:
    print("\n" + "="*80)
    print("CREATING SHORTAGE COMPARISONS FOR EACH RUN")
    print("="*80)

    # Create shortage comparison output directory  
    shortage_comparison_path = join(figure_output_path, "shortage_comparisons")
    if not os.path.exists(shortage_comparison_path):
        os.makedirs(shortage_comparison_path)

    # Extract shortage data from all three datasets (expected/true values) - use filtered data
    original_shortages = original_combined_dict['shortage_data']  # True shortage values: (n_runs, n_timesteps, n_rights)
    wet_shortages = wet_combined_dict['shortage_data']  # True shortage values: (n_runs, n_timesteps, n_rights)
    dry_shortages = dry_combined_dict['shortage_data']  # True shortage values: (n_runs, n_timesteps, n_rights)

    # Convert to numpy if they are torch tensors
    if hasattr(original_shortages, 'numpy'):
        original_shortages = original_shortages.numpy()
    if hasattr(wet_shortages, 'numpy'):
        wet_shortages = wet_shortages.numpy()
    if hasattr(dry_shortages, 'numpy'):
        dry_shortages = dry_shortages.numpy()

    print(f"Original shortages shape: {original_shortages.shape}")
    print(f"Wet shortages shape: {wet_shortages.shape}")
    print(f"Dry shortages shape: {dry_shortages.shape}")

    n_runs_shortage = original_shortages.shape[0]
    print(f"Creating {n_runs_shortage} shortage comparison plots...")

    for run_idx in range(3):
        print(f"  Creating shortage plot for run {run_idx + 1}...")
        
        # Extract shortage data for this run
        original_run = original_shortages[run_idx]  # Shape: (n_timesteps, n_rights)
        wet_run = wet_shortages[run_idx]  # Shape: (n_timesteps, n_rights)
        dry_run = dry_shortages[run_idx]  # Shape: (n_timesteps, n_rights)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Run {run_idx + 1}: Water Shortage Comparison (Original vs Wet vs Dry)', 
                    fontsize=16, fontweight='bold')
        
        # Find common color scale for consistency
        vmin = min(original_run.min(), wet_run.min(), dry_run.min())
        vmax = max(original_run.max(), wet_run.max(), dry_run.max())
        
        # Plot 1: Original shortages
        im1 = axes[0].imshow(original_run.T, aspect='auto', cmap='Reds', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Original Shortages', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Steps', fontsize=12)
        axes[0].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im1, ax=axes[0], label='Shortage Ratio')
        
        # Plot 2: Wet shortages
        im2 = axes[1].imshow(wet_run.T, aspect='auto', cmap='Reds', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title('Wet Shortages (Increasing Streamflow)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Steps', fontsize=12)
        axes[1].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im2, ax=axes[1], label='Shortage Ratio')
        
        # Plot 3: Dry shortages
        im3 = axes[2].imshow(dry_run.T, aspect='auto', cmap='Reds', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[2].set_title('Dry Shortages (Decreasing Streamflow)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time Steps', fontsize=12)
        axes[2].set_ylabel('Water Rights', fontsize=12)
        plt.colorbar(im3, ax=axes[2], label='Shortage Ratio')
        
        # Add summary statistics
        original_mean = np.mean(original_run)
        wet_mean = np.mean(wet_run)
        dry_mean = np.mean(dry_run)
        
        fig.text(0.02, 0.02, f'Mean Shortage Ratio - Original: {original_mean:.4f} | Wet: {wet_mean:.4f} | Dry: {dry_mean:.4f}', 
                fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(join(shortage_comparison_path, f"shortage_run_{run_idx + 1}_comparison.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Shortage comparison plots saved to: {shortage_comparison_path}")
    else:
        print("\nSkipping shortage comparison plots (generate_extra_plots = False)")

    ## Create prediction comparison plots across all three datasets ##
    if generate_extra_plots:
        print("\n" + "="*80)
        print("CREATING PREDICTION COMPARISON PLOTS")
        print("="*80)

    # Create prediction comparison output directory
    prediction_comparison_path = join(figure_output_path, "prediction_comparisons")
    if not os.path.exists(prediction_comparison_path):
        os.makedirs(prediction_comparison_path)

    # Extract predictions for all three datasets
    original_predictions = original_results_tuple[2]  # Shape: (n_runs, n_timesteps, n_rights)
    wet_predictions = wet_results_tuple[2]
    dry_predictions = dry_results_tuple[2]

    # Convert to numpy if they are torch tensors
    if hasattr(original_predictions, 'numpy'):
        original_predictions = original_predictions.numpy()
    if hasattr(wet_predictions, 'numpy'):
        wet_predictions = wet_predictions.numpy()
    if hasattr(dry_predictions, 'numpy'):
        dry_predictions = dry_predictions.numpy()

    print(f"Original predictions shape: {original_predictions.shape}")
    print(f"Wet predictions shape: {wet_predictions.shape}")
    print(f"Dry predictions shape: {dry_predictions.shape}")

    # 1. Individual run prediction comparison plots (3-panel: Original, Wet, Dry)
    print("Creating individual run prediction comparison plots...")

    for run_idx in range(3):  # Process all 10 runs
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Find common color scale across all three predictions for this run
        vmin = min(original_predictions[run_idx].min(), wet_predictions[run_idx].min(), dry_predictions[run_idx].min())
        vmax = max(original_predictions[run_idx].max(), wet_predictions[run_idx].max(), dry_predictions[run_idx].max())
        
        # Original predictions
        im1 = axes[0].imshow(original_predictions[run_idx].T, aspect='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title(f'Original Predictions\nRun {run_idx+1}', fontweight='bold')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Water Rights')
        plt.colorbar(im1, ax=axes[0], label='Predicted Shortage Ratio')
        
        # Wet predictions
        im2 = axes[1].imshow(wet_predictions[run_idx].T, aspect='auto', cmap='viridis',
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title(f'Wet Predictions\nRun {run_idx+1}', fontweight='bold')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Water Rights')
        plt.colorbar(im2, ax=axes[1], label='Predicted Shortage Ratio')
        
        # Dry predictions
        im3 = axes[2].imshow(dry_predictions[run_idx].T, aspect='auto', cmap='viridis',
                            vmin=vmin, vmax=vmax, origin='lower')
        axes[2].set_title(f'Dry Predictions\nRun {run_idx+1}', fontweight='bold')
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Water Rights')
        plt.colorbar(im3, ax=axes[2], label='Predicted Shortage Ratio')
        
        # Add summary statistics
        fig.suptitle(f'ML Prediction Comparison - Run {run_idx+1}', fontsize=16, fontweight='bold')
        fig.text(0.02, 0.02, f'Mean Predicted Shortage - Original: {np.mean(original_predictions[run_idx]):.4f} | '
                            f'Wet: {np.mean(wet_predictions[run_idx]):.4f} | Dry: {np.mean(dry_predictions[run_idx]):.4f}', 
                fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(join(prediction_comparison_path, f"predictions_run_{run_idx+1}_comparison.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Average prediction comparison plot (across all runs)
    print("Creating average prediction comparison plot...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Calculate averages across all runs
    original_avg_pred = np.mean(original_predictions, axis=0)
    wet_avg_pred = np.mean(wet_predictions, axis=0)
    dry_avg_pred = np.mean(dry_predictions, axis=0)

    # Find common color scale
    vmin = min(original_avg_pred.min(), wet_avg_pred.min(), dry_avg_pred.min())
    vmax = max(original_avg_pred.max(), wet_avg_pred.max(), dry_avg_pred.max())

    # Original average predictions
    im1 = axes[0].imshow(original_avg_pred.T, aspect='auto', cmap='viridis', 
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Average Original Predictions\n(Across All Runs)', fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Water Rights')
    plt.colorbar(im1, ax=axes[0], label='Predicted Shortage Ratio')

    # Wet average predictions
    im2 = axes[1].imshow(wet_avg_pred.T, aspect='auto', cmap='viridis',
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Average Wet Predictions\n(Across All Runs)', fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Water Rights')
    plt.colorbar(im2, ax=axes[1], label='Predicted Shortage Ratio')

    # Dry average predictions
    im3 = axes[2].imshow(dry_avg_pred.T, aspect='auto', cmap='viridis',
                        vmin=vmin, vmax=vmax, origin='lower')
    axes[2].set_title('Average Dry Predictions\n(Across All Runs)', fontweight='bold')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Water Rights')
    plt.colorbar(im3, ax=axes[2], label='Predicted Shortage Ratio')

    # Add summary statistics
    fig.suptitle('Average ML Predictions Comparison', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.02, f'Mean Predicted Shortage - Original: {np.mean(original_avg_pred):.4f} | '
                        f'Wet: {np.mean(wet_avg_pred):.4f} | Dry: {np.mean(dry_avg_pred):.4f}', 
            fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(join(prediction_comparison_path, "average_predictions_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Prediction difference plots (similar to what we did for streamflow/shortage differences)
    print("Creating prediction difference plots...")

    for run_idx in range(3):  # Process all 10 runs
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Wet - Original prediction difference
        wet_orig_pred_diff = wet_predictions[run_idx] - original_predictions[run_idx]
        im1 = axes[0].imshow(wet_orig_pred_diff.T, aspect='auto', cmap='RdBu_r',
                            vmin=-np.max(np.abs(wet_orig_pred_diff)), vmax=np.max(np.abs(wet_orig_pred_diff)),
                            origin='lower')
        axes[0].set_title(f'Wet - Original Predictions\nRun {run_idx+1}', fontweight='bold')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Water Rights')
        plt.colorbar(im1, ax=axes[0], label='Prediction Difference')
        
        # Dry - Original prediction difference
        dry_orig_pred_diff = dry_predictions[run_idx] - original_predictions[run_idx]
        im2 = axes[1].imshow(dry_orig_pred_diff.T, aspect='auto', cmap='RdBu_r',
                            vmin=-np.max(np.abs(dry_orig_pred_diff)), vmax=np.max(np.abs(dry_orig_pred_diff)),
                            origin='lower')
        axes[1].set_title(f'Dry - Original Predictions\nRun {run_idx+1}', fontweight='bold')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Water Rights')
        plt.colorbar(im2, ax=axes[1], label='Prediction Difference')
        
        # Wet - Dry prediction difference
        wet_dry_pred_diff = wet_predictions[run_idx] - dry_predictions[run_idx]
        im3 = axes[2].imshow(wet_dry_pred_diff.T, aspect='auto', cmap='RdBu_r',
                            vmin=-np.max(np.abs(wet_dry_pred_diff)), vmax=np.max(np.abs(wet_dry_pred_diff)),
                            origin='lower')
        axes[2].set_title(f'Wet - Dry Predictions\nRun {run_idx+1}', fontweight='bold')
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Water Rights')
        plt.colorbar(im3, ax=axes[2], label='Prediction Difference')
        
        fig.suptitle(f'ML Prediction Differences - Run {run_idx+1}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(join(prediction_comparison_path, f"prediction_differences_run_{run_idx+1}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Average prediction difference plots
    print("Creating average prediction difference plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Average prediction differences across all runs
    avg_wet_orig_pred_diff = np.mean(wet_predictions - original_predictions, axis=0)
    avg_dry_orig_pred_diff = np.mean(dry_predictions - original_predictions, axis=0)
    avg_wet_dry_pred_diff = np.mean(wet_predictions - dry_predictions, axis=0)

    # Wet - Original average prediction difference
    im1 = axes[0].imshow(avg_wet_orig_pred_diff.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(avg_wet_orig_pred_diff)), vmax=np.max(np.abs(avg_wet_orig_pred_diff)),
                        origin='lower')
    axes[0].set_title('Average Wet - Original Predictions\n(Across All Runs)', fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Water Rights')
    plt.colorbar(im1, ax=axes[0], label='Prediction Difference')

    # Dry - Original average prediction difference
    im2 = axes[1].imshow(avg_dry_orig_pred_diff.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(avg_dry_orig_pred_diff)), vmax=np.max(np.abs(avg_dry_orig_pred_diff)),
                        origin='lower')
    axes[1].set_title('Average Dry - Original Predictions\n(Across All Runs)', fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Water Rights')
    plt.colorbar(im2, ax=axes[1], label='Prediction Difference')

    # Wet - Dry average prediction difference
    im3 = axes[2].imshow(avg_wet_dry_pred_diff.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(avg_wet_dry_pred_diff)), vmax=np.max(np.abs(avg_wet_dry_pred_diff)),
                        origin='lower')
    axes[2].set_title('Average Wet - Dry Predictions\n(Across All Runs)', fontweight='bold')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Water Rights')
    plt.colorbar(im3, ax=axes[2], label='Prediction Difference')

    fig.suptitle('Average ML Prediction Differences', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(join(prediction_comparison_path, "average_prediction_differences.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Prediction statistics time series plots
    print("Creating prediction statistics time series plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate statistics over time (average across water rights for each time step)
    original_pred_timeseries = np.mean(original_predictions, axis=(0, 2))  # Average across runs and rights
    wet_pred_timeseries = np.mean(wet_predictions, axis=(0, 2))
    dry_pred_timeseries = np.mean(dry_predictions, axis=(0, 2))

    time_steps = range(len(original_pred_timeseries))

    # Plot 1: Mean predictions over time
    axes[0,0].plot(time_steps, original_pred_timeseries, 'k-', linewidth=2, label='Original', alpha=0.8)
    axes[0,0].plot(time_steps, wet_pred_timeseries, 'b-', linewidth=2, label='Wet', alpha=0.8)
    axes[0,0].plot(time_steps, dry_pred_timeseries, 'r-', linewidth=2, label='Dry', alpha=0.8)
    axes[0,0].set_title('Mean Predicted Shortage Over Time', fontweight='bold')
    axes[0,0].set_xlabel('Time Steps')
    axes[0,0].set_ylabel('Mean Predicted Shortage Ratio')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Plot 2: Standard deviation of predictions over time
    original_pred_std = np.std(original_predictions, axis=(0, 2))
    wet_pred_std = np.std(wet_predictions, axis=(0, 2))
    dry_pred_std = np.std(dry_predictions, axis=(0, 2))

    axes[0,1].plot(time_steps, original_pred_std, 'k-', linewidth=2, label='Original', alpha=0.8)
    axes[0,1].plot(time_steps, wet_pred_std, 'b-', linewidth=2, label='Wet', alpha=0.8)
    axes[0,1].plot(time_steps, dry_pred_std, 'r-', linewidth=2, label='Dry', alpha=0.8)
    axes[0,1].set_title('Std Dev of Predicted Shortage Over Time', fontweight='bold')
    axes[0,1].set_xlabel('Time Steps')
    axes[0,1].set_ylabel('Std Dev of Predicted Shortage Ratio')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Prediction differences over time
    axes[1,0].plot(time_steps, wet_pred_timeseries - original_pred_timeseries, 'b-', linewidth=2, label='Wet - Original', alpha=0.8)
    axes[1,0].plot(time_steps, dry_pred_timeseries - original_pred_timeseries, 'r-', linewidth=2, label='Dry - Original', alpha=0.8)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].set_title('Prediction Differences Over Time', fontweight='bold')
    axes[1,0].set_xlabel('Time Steps')
    axes[1,0].set_ylabel('Difference in Mean Predicted Shortage')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Cumulative prediction differences
    cumsum_wet_diff = np.cumsum(wet_pred_timeseries - original_pred_timeseries)
    cumsum_dry_diff = np.cumsum(dry_pred_timeseries - original_pred_timeseries)

    axes[1,1].plot(time_steps, cumsum_wet_diff, 'b-', linewidth=2, label='Cumulative Wet - Original', alpha=0.8)
    axes[1,1].plot(time_steps, cumsum_dry_diff, 'r-', linewidth=2, label='Cumulative Dry - Original', alpha=0.8)
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_title('Cumulative Prediction Differences', fontweight='bold')
    axes[1,1].set_xlabel('Time Steps')
    axes[1,1].set_ylabel('Cumulative Difference')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(join(prediction_comparison_path, "prediction_statistics_timeseries.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Prediction comparison plots saved to: {prediction_comparison_path}")
    print("Generated prediction comparison files:")
    print("  - predictions_run_X_comparison.png: Individual run 3-panel prediction comparisons")
    print("  - average_predictions_comparison.png: Average predictions across all runs")
    print("  - prediction_differences_run_X.png: Individual run prediction differences")
    print("  - average_prediction_differences.png: Average prediction differences")
    print("  - prediction_statistics_timeseries.png: Time series statistics of predictions")
else:
    print("\nSkipping prediction comparison plots (generate_extra_plots = False)")

## Quantify and verify differences between datasets ##
print("\n" + "="*80)
print("QUANTIFYING DIFFERENCES BETWEEN DATASETS")
print("="*80)

# Create quantification output directory
quantification_path = join(figure_output_path, "difference_quantification")
if not os.path.exists(quantification_path):
    os.makedirs(quantification_path)

print("\n--- STREAMFLOW DIFFERENCES ---")

# Calculate overall streamflow statistics
original_streamflow_mean = np.mean(original_streamflow)
wet_streamflow_mean = np.mean(wet_streamflow)
dry_streamflow_mean = np.mean(dry_streamflow)

original_streamflow_std = np.std(original_streamflow)
wet_streamflow_std = np.std(wet_streamflow)
dry_streamflow_std = np.std(dry_streamflow)

print(f"Original streamflow - Mean: {original_streamflow_mean:.2f}, Std: {original_streamflow_std:.2f}")
print(f"Wet streamflow - Mean: {wet_streamflow_mean:.2f}, Std: {wet_streamflow_std:.2f}")
print(f"Dry streamflow - Mean: {dry_streamflow_mean:.2f}, Std: {dry_streamflow_std:.2f}")

# Calculate percentage differences
wet_streamflow_pct_change = ((wet_streamflow_mean - original_streamflow_mean) / original_streamflow_mean) * 100
dry_streamflow_pct_change = ((dry_streamflow_mean - original_streamflow_mean) / original_streamflow_mean) * 100

print(f"\nStreamflow percentage changes:")
print(f"Wet vs Original: {wet_streamflow_pct_change:.2f}%")
print(f"Dry vs Original: {dry_streamflow_pct_change:.2f}%")

# Calculate element-wise differences
wet_minus_original_streamflow = wet_streamflow - original_streamflow
dry_minus_original_streamflow = dry_streamflow - original_streamflow

print(f"\nStreamflow absolute differences (mean ± std):")
print(f"Wet - Original: {np.mean(wet_minus_original_streamflow):.2f} ± {np.std(wet_minus_original_streamflow):.2f}")
print(f"Dry - Original: {np.mean(dry_minus_original_streamflow):.2f} ± {np.std(dry_minus_original_streamflow):.2f}")

print(f"\nStreamflow relative differences (as % of original):")
wet_relative_diff = (wet_minus_original_streamflow / (original_streamflow + 1e-10)) * 100  # Add small epsilon to avoid division by zero
dry_relative_diff = (dry_minus_original_streamflow / (original_streamflow + 1e-10)) * 100

print(f"Wet relative change: {np.mean(wet_relative_diff):.2f}% ± {np.std(wet_relative_diff):.2f}%")
print(f"Dry relative change: {np.mean(dry_relative_diff):.2f}% ± {np.std(dry_relative_diff):.2f}%")

print("\n--- SHORTAGE DIFFERENCES ---")

# Calculate overall shortage statistics
original_shortage_mean = np.mean(original_shortages)
wet_shortage_mean = np.mean(wet_shortages)
dry_shortage_mean = np.mean(dry_shortages)

original_shortage_std = np.std(original_shortages)
wet_shortage_std = np.std(wet_shortages)
dry_shortage_std = np.std(dry_shortages)

print(f"Original shortages - Mean: {original_shortage_mean:.4f}, Std: {original_shortage_std:.4f}")
print(f"Wet shortages - Mean: {wet_shortage_mean:.4f}, Std: {wet_shortage_std:.4f}")
print(f"Dry shortages - Mean: {dry_shortage_mean:.4f}, Std: {dry_shortage_std:.4f}")

# Calculate percentage differences
wet_shortage_pct_change = ((wet_shortage_mean - original_shortage_mean) / (original_shortage_mean + 1e-10)) * 100
dry_shortage_pct_change = ((dry_shortage_mean - original_shortage_mean) / (original_shortage_mean + 1e-10)) * 100

print(f"\nShortage percentage changes:")
print(f"Wet vs Original: {wet_shortage_pct_change:.2f}%")
print(f"Dry vs Original: {dry_shortage_pct_change:.2f}%")

# Calculate element-wise differences
wet_minus_original_shortage = wet_shortages - original_shortages
dry_minus_original_shortage = dry_shortages - original_shortages

print(f"\nShortage absolute differences (mean ± std):")
print(f"Wet - Original: {np.mean(wet_minus_original_shortage):.4f} ± {np.std(wet_minus_original_shortage):.4f}")
print(f"Dry - Original: {np.mean(dry_minus_original_shortage):.4f} ± {np.std(dry_minus_original_shortage):.4f}")

# Create difference histogram plots
print("\nCreating difference distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribution of Differences Between Scenarios', fontsize=16, fontweight='bold')

# Streamflow difference histograms
axes[0,0].hist(wet_minus_original_streamflow.flatten(), bins=50, alpha=0.7, color='blue', density=True)
axes[0,0].axvline(np.mean(wet_minus_original_streamflow), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(wet_minus_original_streamflow):.1f}')
axes[0,0].set_title('Wet - Original Streamflow Differences', fontweight='bold')
axes[0,0].set_xlabel('Difference (acre-feet)')
axes[0,0].set_ylabel('Density')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].hist(dry_minus_original_streamflow.flatten(), bins=50, alpha=0.7, color='orange', density=True)
axes[0,1].axvline(np.mean(dry_minus_original_streamflow), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dry_minus_original_streamflow):.1f}')
axes[0,1].set_title('Dry - Original Streamflow Differences', fontweight='bold')
axes[0,1].set_xlabel('Difference (acre-feet)')
axes[0,1].set_ylabel('Density')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Shortage difference histograms
axes[1,0].hist(wet_minus_original_shortage.flatten(), bins=50, alpha=0.7, color='blue', density=True)
axes[1,0].axvline(np.mean(wet_minus_original_shortage), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(wet_minus_original_shortage):.4f}')
axes[1,0].set_title('Wet - Original Shortage Differences', fontweight='bold')
axes[1,0].set_xlabel('Difference (shortage ratio)')
axes[1,0].set_ylabel('Density')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].hist(dry_minus_original_shortage.flatten(), bins=50, alpha=0.7, color='orange', density=True)
axes[1,1].axvline(np.mean(dry_minus_original_shortage), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dry_minus_original_shortage):.4f}')
axes[1,1].set_title('Dry - Original Shortage Differences', fontweight='bold')
axes[1,1].set_xlabel('Difference (shortage ratio)')
axes[1,1].set_ylabel('Density')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(join(quantification_path, "difference_distributions.png"), dpi=300, bbox_inches='tight')
plt.close()

# Create time series of differences for a specific run and gauge
print("Creating time series difference plots...")

# Pick run 0 and gauge 0 for detailed inspection
run_to_plot = 0
gauge_to_plot = 20  # INK20000 outflow gauge index

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle(f'Time Series Differences - Run {run_to_plot+1}, Gauge {gauge_to_plot}', fontsize=16, fontweight='bold')

# Plot streamflow differences over time
time_steps = range(original_streamflow.shape[1])
axes[0].plot(time_steps, wet_streamflow[run_to_plot, :, gauge_to_plot] - original_streamflow[run_to_plot, :, gauge_to_plot], 
            'b-', linewidth=2, label='Wet - Original', alpha=0.8)
axes[0].plot(time_steps, dry_streamflow[run_to_plot, :, gauge_to_plot] - original_streamflow[run_to_plot, :, gauge_to_plot], 
            'r-', linewidth=2, label='Dry - Original', alpha=0.8)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0].set_title('Streamflow Differences Over Time', fontweight='bold')
axes[0].set_xlabel('Time Steps')
axes[0].set_ylabel('Streamflow Difference (acre-feet)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot shortage differences over time (average across all rights)
axes[1].plot(time_steps, np.mean(wet_shortages[run_to_plot, :, :] - original_shortages[run_to_plot, :, :], axis=1), 
            'b-', linewidth=2, label='Wet - Original (avg across rights)', alpha=0.8)
axes[1].plot(time_steps, np.mean(dry_shortages[run_to_plot, :, :] - original_shortages[run_to_plot, :, :], axis=1), 
            'r-', linewidth=2, label='Dry - Original (avg across rights)', alpha=0.8)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].set_title('Average Shortage Differences Over Time', fontweight='bold')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('Average Shortage Difference')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(join(quantification_path, "time_series_differences.png"), dpi=300, bbox_inches='tight')
plt.close()

## Create difference imshow plots for dataset pairs ##
print("\nCreating difference imshow plots...")

# For each run, create imshow plots showing differences between dataset pairs
for run_idx in range(3):  # Process all 10 runs
    
    # STREAMFLOW DIFFERENCES IMSHOW
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Wet - Original streamflow difference
    wet_orig_diff = wet_streamflow[run_idx, :, :] - original_streamflow[run_idx, :, :]
    im1 = axes[0].imshow(wet_orig_diff.T, aspect='auto', cmap='RdBu_r', 
                        vmin=-np.max(np.abs(wet_orig_diff)), vmax=np.max(np.abs(wet_orig_diff)))
    axes[0].set_title(f'Wet - Original Streamflow\nRun {run_idx+1}', fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Gauges')
    plt.colorbar(im1, ax=axes[0], label='Streamflow Difference (acre-feet)')
    
    # Dry - Original streamflow difference  
    dry_orig_diff = dry_streamflow[run_idx, :, :] - original_streamflow[run_idx, :, :]
    im2 = axes[1].imshow(dry_orig_diff.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(dry_orig_diff)), vmax=np.max(np.abs(dry_orig_diff)))
    axes[1].set_title(f'Dry - Original Streamflow\nRun {run_idx+1}', fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Gauges')
    plt.colorbar(im2, ax=axes[1], label='Streamflow Difference (acre-feet)')
    
    # Wet - Dry streamflow difference
    wet_dry_diff = wet_streamflow[run_idx, :, :] - dry_streamflow[run_idx, :, :]
    im3 = axes[2].imshow(wet_dry_diff.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(wet_dry_diff)), vmax=np.max(np.abs(wet_dry_diff)))
    axes[2].set_title(f'Wet - Dry Streamflow\nRun {run_idx+1}', fontweight='bold')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Gauges')
    plt.colorbar(im3, ax=axes[2], label='Streamflow Difference (acre-feet)')
    
    plt.tight_layout()
    plt.savefig(join(quantification_path, f"streamflow_differences_run_{run_idx+1}_imshow.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHORTAGE DIFFERENCES IMSHOW
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Wet - Original shortage difference
    wet_orig_shortage_diff = wet_shortages[run_idx, :, :] - original_shortages[run_idx, :, :]
    im1 = axes[0].imshow(wet_orig_shortage_diff.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(wet_orig_shortage_diff)), vmax=np.max(np.abs(wet_orig_shortage_diff)))
    axes[0].set_title(f'Wet - Original Shortages\nRun {run_idx+1}', fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Water Rights')
    plt.colorbar(im1, ax=axes[0], label='Shortage Difference')
    
    # Dry - Original shortage difference
    dry_orig_shortage_diff = dry_shortages[run_idx, :, :] - original_shortages[run_idx, :, :]
    im2 = axes[1].imshow(dry_orig_shortage_diff.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(dry_orig_shortage_diff)), vmax=np.max(np.abs(dry_orig_shortage_diff)))
    axes[1].set_title(f'Dry - Original Shortages\nRun {run_idx+1}', fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Water Rights')
    plt.colorbar(im2, ax=axes[1], label='Shortage Difference')
    
    # Wet - Dry shortage difference
    wet_dry_shortage_diff = wet_shortages[run_idx, :, :] - dry_shortages[run_idx, :, :]
    im3 = axes[2].imshow(wet_dry_shortage_diff.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(wet_dry_shortage_diff)), vmax=np.max(np.abs(wet_dry_shortage_diff)))
    axes[2].set_title(f'Wet - Dry Shortages\nRun {run_idx+1}', fontweight='bold')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Water Rights')
    plt.colorbar(im3, ax=axes[2], label='Shortage Difference')
    
    plt.tight_layout()
    plt.savefig(join(quantification_path, f"shortage_differences_run_{run_idx+1}_imshow.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

# Create average difference imshow plots across all runs
print("Creating average difference imshow plots...")

# AVERAGE STREAMFLOW DIFFERENCES IMSHOW
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Average Wet - Original streamflow difference
avg_wet_orig_diff = np.mean(wet_streamflow - original_streamflow, axis=0)
im1 = axes[0].imshow(avg_wet_orig_diff.T, aspect='auto', cmap='RdBu_r',
                    vmin=-np.max(np.abs(avg_wet_orig_diff)), vmax=np.max(np.abs(avg_wet_orig_diff)))
axes[0].set_title('Average Wet - Original Streamflow\n(Across All Runs)', fontweight='bold')
axes[0].set_xlabel('Time Steps')
axes[0].set_ylabel('Gauges')
plt.colorbar(im1, ax=axes[0], label='Streamflow Difference (acre-feet)')

# Average Dry - Original streamflow difference
avg_dry_orig_diff = np.mean(dry_streamflow - original_streamflow, axis=0)
im2 = axes[1].imshow(avg_dry_orig_diff.T, aspect='auto', cmap='RdBu_r',
                    vmin=-np.max(np.abs(avg_dry_orig_diff)), vmax=np.max(np.abs(avg_dry_orig_diff)))
axes[1].set_title('Average Dry - Original Streamflow\n(Across All Runs)', fontweight='bold')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('Gauges')
plt.colorbar(im2, ax=axes[1], label='Streamflow Difference (acre-feet)')

# Average Wet - Dry streamflow difference
avg_wet_dry_diff = np.mean(wet_streamflow - dry_streamflow, axis=0)
im3 = axes[2].imshow(avg_wet_dry_diff.T, aspect='auto', cmap='RdBu_r',
                    vmin=-np.max(np.abs(avg_wet_dry_diff)), vmax=np.max(np.abs(avg_wet_dry_diff)))
axes[2].set_title('Average Wet - Dry Streamflow\n(Across All Runs)', fontweight='bold')
axes[2].set_xlabel('Time Steps')
axes[2].set_ylabel('Gauges')
plt.colorbar(im3, ax=axes[2], label='Streamflow Difference (acre-feet)')

plt.tight_layout()
plt.savefig(join(quantification_path, "average_streamflow_differences_imshow.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# AVERAGE SHORTAGE DIFFERENCES IMSHOW
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Average Wet - Original shortage difference
avg_wet_orig_shortage_diff = np.mean(wet_shortages - original_shortages, axis=0)
im1 = axes[0].imshow(avg_wet_orig_shortage_diff.T, aspect='auto', cmap='RdBu_r',
                    vmin=-np.max(np.abs(avg_wet_orig_shortage_diff)), vmax=np.max(np.abs(avg_wet_orig_shortage_diff)))
axes[0].set_title('Average Wet - Original Shortages\n(Across All Runs)', fontweight='bold')
axes[0].set_xlabel('Time Steps')
axes[0].set_ylabel('Water Rights')
plt.colorbar(im1, ax=axes[0], label='Shortage Difference')

# Average Dry - Original shortage difference
avg_dry_orig_shortage_diff = np.mean(dry_shortages - original_shortages, axis=0)
im2 = axes[1].imshow(avg_dry_orig_shortage_diff.T, aspect='auto', cmap='RdBu_r',
                    vmin=-np.max(np.abs(avg_dry_orig_shortage_diff)), vmax=np.max(np.abs(avg_dry_orig_shortage_diff)))
axes[1].set_title('Average Dry - Original Shortages\n(Across All Runs)', fontweight='bold')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('Water Rights')
plt.colorbar(im2, ax=axes[1], label='Shortage Difference')

# Average Wet - Dry shortage difference
avg_wet_dry_shortage_diff = np.mean(wet_shortages - dry_shortages, axis=0)
im3 = axes[2].imshow(avg_wet_dry_shortage_diff.T, aspect='auto', cmap='RdBu_r',
                    vmin=-np.max(np.abs(avg_wet_dry_shortage_diff)), vmax=np.max(np.abs(avg_wet_dry_shortage_diff)))
axes[2].set_title('Average Wet - Dry Shortages\n(Across All Runs)', fontweight='bold')
axes[2].set_xlabel('Time Steps')
axes[2].set_ylabel('Water Rights')
plt.colorbar(im3, ax=axes[2], label='Shortage Difference')

plt.tight_layout()
plt.savefig(join(quantification_path, "average_shortage_differences_imshow.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

print("Difference imshow plots completed!")

# Save quantification results to CSV
print("Saving quantification results to CSV...")

quantification_results = {
    'Dataset': ['Original', 'Wet', 'Dry'],
    'Streamflow_Mean': [original_streamflow_mean, wet_streamflow_mean, dry_streamflow_mean],
    'Streamflow_Std': [original_streamflow_std, wet_streamflow_std, dry_streamflow_std],
    'Streamflow_PctChange_vs_Original': [0.0, wet_streamflow_pct_change, dry_streamflow_pct_change],
    'Shortage_Mean': [original_shortage_mean, wet_shortage_mean, dry_shortage_mean],
    'Shortage_Std': [original_shortage_std, wet_shortage_std, dry_shortage_std],
    'Shortage_PctChange_vs_Original': [0.0, wet_shortage_pct_change, dry_shortage_pct_change]
}

quantification_df = pd.DataFrame(quantification_results)
quantification_df.to_csv(join(quantification_path, "dataset_differences_summary.csv"), index=False)

print(f"Quantification results saved to: {quantification_path}")
print("Generated files:")
print("  - difference_distributions.png: Histograms of differences")
print("  - time_series_differences.png: Time series of differences")
print("  - streamflow_differences_run_X_imshow.png: Individual run streamflow difference imshow plots (3-panel)")
print("  - shortage_differences_run_X_imshow.png: Individual run shortage difference imshow plots (3-panel)")
print("  - average_streamflow_differences_imshow.png: Average streamflow differences across all runs")
print("  - average_shortage_differences_imshow.png: Average shortage differences across all runs")
print("  - dataset_differences_summary.csv: Statistical summary")

## Summary ##
print("\n" + "="*80)
print("PREDICTION GENERATION COMPLETE")
print("="*80)

print("Generated predictions for:")
print(f"  1. Original subset dataset ({n_subset} runs): {original_predictions_path}")
print(f"  2. Wet subset dataset ({n_subset} runs): {wet_predictions_path}")
print(f"  3. Dry subset dataset ({n_subset} runs): {dry_predictions_path}")

print(f"\nOriginal predictions shape: {original_results_tuple[2].shape}")
print(f"Wet predictions shape: {wet_results_tuple[2].shape}")
print(f"Dry predictions shape: {dry_results_tuple[2].shape}")

print("\nAnalysis completed:")
print("  - Generated ML predictions for all three scenarios")
print("  - Calculated comprehensive error metrics (MSE, MAE, ME, NSE)")  
print("  - Created boxplot visualizations comparing model performance")
print("  - Produced statistical summaries for detailed analysis")

print("\nKey insights available:")
print("  - How model performance varies across different hydrological conditions")
print("  - Whether the ML model adapts well to non-stationary streamflows")
print("  - Comparative performance on wet vs. dry climate scenarios")
print("  - Both standard and volumetric error metrics for comprehensive evaluation")

print(f"\nVisualization outputs created:")
print(f"  - Error metric boxplots: {figure_output_path}")
print(f"  - ML prediction comparisons: wet_comparisons/, original_comparisons/, dry_comparisons/")
print(f"  - Streamflow comparisons: {streamflow_comparison_path}")
print(f"  - Shortage comparisons: {shortage_comparison_path}")

print(f"\nFiles ready for analysis and comparison plotting!")
print(f"Note: Working with {n_subset} streamflows for faster processing and testing")
print("All three scenarios (original, wet, dry) with complete error analysis are ready!")

print(f"\nGenerated plot types:")
print(f"  1. Boxplots: Model performance metrics across scenarios")
print(f"  2. ML Prediction Comparisons: Expected vs Predicted for each dataset")
print(f"  3. Streamflow Comparisons: Original vs Wet vs Dry streamflow patterns")
print(f"  4. Shortage Comparisons: Original vs Wet vs Dry shortage patterns")

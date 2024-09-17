
from os.path import join
import torch
import optuna
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
import json
from toolkit.emulator.models import BatchWaterLSTM
from toolkit.utils.io import hdf5_to_dict
from toolkit.emulator.dataset import WrapDataset
from toolkit import outputs_data_path


## Settings ##
random_seed = 42
n_trials = 1000

## Path configuration ##
optimized_params_path = join(outputs_data_path, "optimized_params.json")
dataset_path = join(
    outputs_data_path,
    "synthetic-trainvalid",
    "synthetic_trainvalid_dataset.h5")

## Main Script ##

# load data
dataset = WrapDataset(hdf5_to_dict(dataset_path))
generator = torch.Generator().manual_seed(random_seed)
train_data, valid_data = random_split(dataset, [0.8,0.2], generator)

## Hyperparameter Optimization with Optuna ##
input_dim = dataset.data_dict["streamflow_data"].shape[2]
output_dim = dataset.data_dict["shortage_data"].shape[2]


# Define objective function
def objective(trial):
    # Define hyperparameters using trial.suggest_*
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    batch_size = trial.suggest_categorical('batch_size', [5, 10, 32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    epochs = 300
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize dataloaders with batch size choice
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    best_val_loss = 10000
    # Define model, loss function, and optimizer
    print(
        f"input_dim={input_dim}", 
        f"hidden_dim={hidden_dim}", 
        f"output_dim={output_dim}",
        f"dropout={dropout}",
        f"num_layers={num_layers}")
    model = BatchWaterLSTM(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim,
        dropout=dropout,
        num_layers=num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        print(epoch)
        step_loss = []
        model.train()
        for i, (streamflows, diversions) in enumerate(train_data_loader):
            streamflows = streamflows.to(device)
            diversions = diversions.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(streamflows)
            train_loss = criterion(outputs, diversions)
            train_loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        step_loss = []
        for i, (streamflows, diversions) in enumerate(valid_data_loader):
            streamflows = streamflows.to(device)
            diversions = diversions.to(device)
            # Forward Pass
            outputs = model(streamflows)
            # Find the Loss
            test_loss = criterion(outputs, diversions)
            # Calculate Loss
            step_loss.append(test_loss.item())
        val_loss = np.array(step_loss).mean()

        # Report intermediate objective value.
        trial.report(val_loss, epoch)

        print(val_loss, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
    return best_val_loss  # Return the metric to optimize

sampler = optuna.samplers.TPESampler(seed=random_seed)  # Make the sampler behave in a deterministic way.
study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective, n_trials=n_trials)

# save best params
with open(optimized_params_path, 'w') as json_file:
    json.dump(study.best_trial.params, json_file, indent=4)
    
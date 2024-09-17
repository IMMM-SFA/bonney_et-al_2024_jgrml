from os.path import join
import torch
from torch.utils.data import DataLoader, random_split
from toolkit.utils.io import hdf5_to_dict, load_config
from toolkit.emulator.dataset import WrapDataset
from toolkit import repo_data_path, outputs_data_path
from toolkit.emulator.trainer import Trainer


## Settings ##

# set random seed for reproductibility
random_seed = 8888

## Path Configurations ##

# config file
config_file = join(repo_data_path, "ml-configs", "optimized_params.yaml")

# training dataset
ml_dataset_path = join(outputs_data_path, "synthetic-trainvalid", "synthetic_trainvalid_dataset.h5")

# model output directory
model_output_path = join(outputs_data_path, "runs")

## Main Script ##

# Load and adjust config
config = load_config(config_file)
config["model"]["name"] = "BatchWaterLSTM"

config["data"]["data_path"] = ml_dataset_path
config["random_seed"] = random_seed

# apply random seed
torch.manual_seed(random_seed)
generator = torch.Generator().manual_seed(random_seed)

# Load Dataset and create loaders
dataset = WrapDataset(hdf5_to_dict(ml_dataset_path))

train_data, valid_data = random_split(dataset, [0.8,0.2], generator=generator)

data_config = config["data"]
train_data_loader = DataLoader(train_data, batch_size=data_config["batch_size"], shuffle=True)
valid_data_loader = DataLoader(valid_data, batch_size=data_config["batch_size"], shuffle=True)

# load example data to determine model shape
sample_input, sample_target = train_data_loader.dataset[0]
data_config["input_time_steps"] = sample_input.shape[0]
data_config["input_dim"] = sample_input.shape[1]  # number of sites
data_config["output_dim"] = sample_target.shape[1]  # number of rights

# fetch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# verify device
assert device != "cpu" # remove this if you intend to run on a cpu (not recommended)

# Initialize, build, and train model
trainer = Trainer(device, output_dir=model_output_path)
trainer.build(config)
trainer.train(train_data_loader, valid_data_loader=valid_data_loader)
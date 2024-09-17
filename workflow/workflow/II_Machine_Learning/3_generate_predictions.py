from os.path import join
import torch
from torch.utils.data import DataLoader, SequentialSampler
from toolkit.emulator.dataset import WrapDataset
from toolkit.utils.io import hdf5_to_dict, load_config, dict_to_hdf5
from toolkit import outputs_data_path
from toolkit.emulator.trainer import Trainer


## Settings ##

# load in model
# select model run and checkpoint
run_name = "run_20240916-123332" # <---- Replace this with the directory name in runs/ containing the trained model of interest
checkpoint_id = "249" # this is the checkpoint used in the paper and should not be changed if attempting to reproduce results.

## Path Configuration ##

model_dir = join(outputs_data_path, "runs", run_name)
# config path in model direcotry
config_file = join(model_dir, "config.yaml")

synthetic_test_data_path = join(outputs_data_path,
    "synthetic-test")

## Main Script ##

# Load config
config = load_config(config_file)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(device, output_dir=model_dir)
trainer.build(config)
trainer.load_checkpoint(checkpoint_id)

# Load test data and generate predictions using trained ML model
for drought in [0.0,0.1,0.2,0.3,0.4,0.5]:        
    dataset_path = join(
        synthetic_test_data_path,
        f"synthetic_test_dataset_drought_{str(drought)}.h5"
    )
    
    # Load test dataset
    dataset = WrapDataset(hdf5_to_dict(dataset_path))
    
    # use sequential sampler and no shuffling for consistent ordering
    test_sampler = SequentialSampler(dataset)
    test_data_loader = DataLoader(dataset, shuffle=False, sampler=test_sampler)
    results_tuple = trainer.get_targets_and_predictions(test_data_loader)
    
    data_dict = {"shortage_predictions": results_tuple[2]}
    filepath = join(
        synthetic_test_data_path,
        f"synthetic_test_dataset_drought_{str(drought)}_{run_name}_predictions.h5")
    dict_to_hdf5(filepath, data_dict)

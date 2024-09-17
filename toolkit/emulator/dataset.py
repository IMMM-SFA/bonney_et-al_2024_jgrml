import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join 
from toolkit import repo_data_path
from torch.utils.data import DataLoader, random_split
from toolkit.utils.io import hdf5_to_dict


class WrapDataset(Dataset):
    def __init__(self, data_dict):
        data_dict["streamflow_data"] = torch.tensor(data_dict["streamflow_data"]).float()
        data_dict["shortage_data"] = torch.tensor(data_dict["shortage_data"]).float()
        self.data_dict = data_dict
        self.len = data_dict["shortage_data"].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        streamflow = self.data_dict["streamflow_data"][idx]
        shortage = self.data_dict["shortage_data"][idx]
        return streamflow, shortage
  
def load_historical_dataset(rights, path=None):
    rights = list(rights)
    if path is None:
        path = join(repo_data_path, "ml-data", "historical", "historical_dataset_1940-2016_1_samples.h5")
    historical_dataset = WrapDataset(hdf5_to_dict(path))
    historical_rights = list(historical_dataset.data_dict["shortage_columns"])

    right_filter = [True if right in rights else False for right in historical_rights]

    historical_dataset.data_dict["shortage_columns"] = historical_dataset.data_dict["shortage_columns"][right_filter]
    historical_dataset.data_dict["shortage_data"] = historical_dataset.data_dict["shortage_data"][:,:,right_filter]
    
    return historical_dataset
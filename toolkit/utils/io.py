import h5py
import pandas as pd
import geopandas as gpd
import numpy as np
from os.path import join
from toolkit import repo_data_path
import yaml

def dict_to_hdf5(filepath, data_dictionary):
    with h5py.File(filepath, "w") as f:
        for key, value in data_dictionary.items():
            f.create_dataset(key, data=value)

def hdf5_to_dict(filepath): 
    data_dict = dict()
    with h5py.File(filepath) as f:
        for key in f.keys():
            if f[key].dtype == "O":
                data_dict[key] = f[key].asstr()[:]
            elif f[key].dtype == "f8" and key=="shortage_data": # special case for shortages
                data_dict[key] = np.array(f[key], dtype=np.float32)
            else:    
                data_dict[key] = f[key][:]
    return data_dict

def load_right_latlongs(latlong_gdf_path=None):
    if latlong_gdf_path is None:
        latlong_gdf_path = join(repo_data_path, "geospatial", "right_latlongs.geojson")
    latlongs = gpd.read_file(latlong_gdf_path)
    latlongs.set_index("water_right_identifier", inplace=True)
    return latlongs

def load_crb_shape(crb_path=None):
    if crb_path is None:
        crb_path = join(repo_data_path, "geospatial", "CRB")
    crb = gpd.read_file(crb_path)
    return crb

def load_config(file):
    with open(file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
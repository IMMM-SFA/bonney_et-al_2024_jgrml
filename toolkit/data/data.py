import numpy as np

def filter_dataset(streamflow_dict, shortage_dict, train_datadict):
    """Takes streamflow and shortage data and filters the columns
    based on out the data in train_datadict were filtered

    Parameters
    ----------
    streamflow_dict : dict
        _description_
    shortage_dict : dict
        _description_
    train_datadict : dict
        _description_

    Returns
    -------
    dict
        _description_
    """
    
    streamflow_data = streamflow_dict["streamflow_data"]
    streamflow_index = streamflow_dict["streamflow_index"]
    streamflow_columns = streamflow_dict["streamflow_columns"]

    shortage_data = shortage_dict["shortage_data"]
    shortage_index = shortage_dict["shortage_index"]
    shortage_columns = shortage_dict["shortage_columns"]

    # 
    train_shortage_index = train_datadict["shortage_index"]
    index_mask = np.isin(shortage_index, train_shortage_index)
    assert (shortage_index[index_mask] == train_shortage_index).all()

    # 
    train_shortage_columns = train_datadict["shortage_columns"]
    column_mask = np.isin(shortage_columns, train_shortage_columns)
    assert (shortage_columns[column_mask] == train_shortage_columns).all()

    # 
    shortage_data = shortage_data[:,index_mask,:]
    shortage_data = shortage_data[:,:,column_mask]
    shortage_index = shortage_index[index_mask]
    shortage_columns = shortage_columns[column_mask]

    # 
    streamflow_data = streamflow_data[:,index_mask,:]
    streamflow_index = streamflow_index[index_mask]

    right_sectors = train_datadict["sector"]
    right_seniority = train_datadict["seniority"]
    right_allotments = train_datadict["allotment"]

    # converting nans
    shortage_data[np.isnan(shortage_data)] = 0.0

    # assert that data structures shapes line up as a final check
    assert streamflow_data.shape[1] == streamflow_index.shape[0]
    assert streamflow_data.shape[2] == streamflow_columns.shape[0]

    assert shortage_data.shape[1] == shortage_index.shape[0]
    assert shortage_data.shape[2] == shortage_columns.shape[0]
    assert shortage_data.shape[2] == right_sectors.shape[0]
    assert shortage_data.shape[2] == right_seniority.shape[0]

    assert shortage_data.shape[0] == streamflow_data.shape[0]
    assert shortage_data.shape[1] == streamflow_data.shape[1]

    assert (streamflow_index == shortage_index).all()


    # organize data back into a dictionary
    data_dict = dict() 
    data_dict["streamflow_data"] = streamflow_data
    data_dict["streamflow_index"] = streamflow_index
    data_dict["streamflow_columns"] = streamflow_columns

    data_dict["shortage_data"] = shortage_data
    data_dict["shortage_index"] = shortage_index
    data_dict["shortage_columns"] = shortage_columns
    data_dict["sector"] = right_sectors
    data_dict["seniority"] = right_seniority
    data_dict["allotment"] = right_allotments

    print("Final streamflow data shape:", streamflow_data.shape)
    print("Final shortage data shape:", shortage_data.shape)

    return data_dict
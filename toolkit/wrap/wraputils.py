import os


def clean_folders(
    execution_directory,
    shortage_directory,
    flo_directory):
    """Removes OUT, MSS, and FLO files from subdirectories

    Parameters
    ----------
    execution_directory : str
        directory that contains multiple directories named "execution_folder_i" 
        where i is some integer.
    """

    # clean execution folder
    for directory in next(os.walk(execution_directory))[1]:
        for file in next(os.walk(os.path.join(execution_directory,directory)))[2]:
            if "OUT" in file or "MSS" in file or "FLO" in file:
                os.remove(os.path.join(execution_directory,directory, file))
                
    # clean shortage folder
    for file in next(os.walk(shortage_directory))[2]:
        if ".csv" in file: 
            os.remove(os.path.join(shortage_directory, file))
                
    # clean flo folder
    for file in next(os.walk(flo_directory))[2]:
        if "FLO" in file: 
            os.remove(os.path.join(flo_directory, file))

def split_into_subslits(lst, n):
    """Basic function to split a list into n sublists of roughly equal size.
    Used to prepare flofile list for multiprocessing.
    """
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    sublists = [lst[i*chunk_size+min(i,remainder):(i+1)*chunk_size+min(i+1,remainder)] for i in range(n)]
    assert [item for sublist in sublists for item in sublist] == lst
    return sublists

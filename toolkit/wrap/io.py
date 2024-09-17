import pandas as pd
from pandas import DataFrame
import numpy as np
from os.path import join
from toolkit.wrap.wrapdata import WRAP_IDS, COL_FLOW_RIGHTS, COL_CONTROL_POINTS, COL_DIVERSION_RIGHTS, COL_RESERVOIR, N_HEADER, COL_WATER_RIGHT
from pathlib import Path
       
    
def df_to_evp(evap_df: DataFrame, file_name: str):
    """Converts a dataframe with evap information to WRAP compatible .EVA file

    Parameters
    ----------
    evap_df : DataFrame
        pandas dataframe with a year column and month column represented as integers.
    file_name : str
        path to .EVA file to be written
    """
    with open(file_name, "wt") as file:
        years = evap_df.index.year.unique()
        sites = [site for site in evap_df.columns if "EV" in site]
        for year in years:
            year_df = evap_df[evap_df.index.year == year]
            for site in sites:
                line = f"{site}{year:>8}"
                for num in year_df[site]:
                    num = float(num)
                    line += f"{num: 8.3f}"
                line += "\n"
                file.write(line)


def df_to_flo(flo_df: DataFrame, filename: str):
    """credit to Stephen/Travis

    Parameters
    ----------
    flo_df : DataFrame
        _description_
    filename : str
        _description_
    """
    IDs = pd.Series(WRAP_IDS)
    stations = flo_df.shape[1]

    # Years of Monthly Data
    start_year = flo_df.index.min().year
    num_years = int(flo_df.shape[0] / 12)

    # Create dataframe to populate with formatted FLO data
    formatted_data = pd.DataFrame(data=np.zeros([stations * num_years, 14]))

    # Format Node ID and Year columns
    Years = list(flo_df.index.year.unique())
    Years_repeating = list(flo_df.index.year)
    CP_col = pd.Series(np.zeros(num_years * stations))
    years_FLO = np.zeros(num_years * stations)
    for i in range(num_years):
        years_FLO[i * stations : (i + 1) * stations] = np.ones(stations) * Years[i]
        CP_col.iloc[i * stations : (i + 1) * stations] = IDs

    formatted_data.iloc[:, 0] = CP_col[:].astype("str")
    formatted_data.iloc[:, 1] = years_FLO[:].astype("int")

    for i in range(12):
        formatted_data.iloc[:, 2 + i] = np.zeros(stations * num_years).astype("int")

    for i in range(num_years):
        for j in range(stations):
            formatted_data.iloc[i * stations + j, 2:14] = (
                flo_df.iloc[i * 12 : (i + 1) * 12, j]
            ).astype("int")

    formatted_data = formatted_data.astype(
        {k: int for k in list(formatted_data.columns) if k != 0}
    )
    lines = []
    for line in range(num_years * stations):
        line = formatted_data.iloc[line, :].astype("str")
        formatted_line = []
        for i in range(14):
            if i == 0:
                formatted_line.append(line[0])
            else:
                padded_entry = line[i].rjust(8)
                formatted_line.append(padded_entry)

        joined_line = [
            formatted_line[0]
            + formatted_line[1]
            + formatted_line[2]
            + formatted_line[3]
            + formatted_line[4]
            + formatted_line[5]
            + formatted_line[6]
            + formatted_line[7]
            + formatted_line[8]
            + formatted_line[9]
            + formatted_line[10]
            + formatted_line[11]
            + formatted_line[12]
            + formatted_line[13]
        ]

        lines.append(joined_line)

    # saves .FLO file with streamflow realization for WRAP input #
    with open(filename, "w") as f:
        for i in range(num_years * stations):
            f.write(str(lines[i][0]))
            f.write("\n")

    return


def evp_to_df(file_name: str, csv_name: str = None):
    """Reads evap file into a pandas dataframe and optionally writes the data as a csv

    Parameters
    ----------
    file_name : str
        path to .EVA file
    csv_name : str, optional
        path to csv file to write to, by default None

    Returns
    -------
    DataFrame
        Evaporation data
    """
    with open(file_name, "rt") as file:
        evap_lines = file.readlines()
    data = []
    for line in evap_lines:
        if line[0] != "*":
            data.append(line.split())
    evap_df = pd.DataFrame(data)
    evap_df = evap_df.dropna()
    evap_df = evap_df.pivot(index=0, columns=1).transpose().swaplevel().sort_index()
    evap_df = evap_df.reset_index()
    evap_df = evap_df.rename(columns={1: "year", "level_1": "month"})
    evap_df["month"] = evap_df["month"] - 1
    evap_df["year"] = evap_df.year.astype(int)
    evap_df.insert(
        0,
        "date",
        evap_df.apply(lambda row: np.datetime64(f"{row.year}-{row.month:02d}"), axis=1),
    )
    evap_df = evap_df.drop(columns=["year", "month"])
    evap_df = evap_df.set_index("date")
    if csv_name:
        evap_df.write_csv(csv_name)

    return evap_df


def flo_to_df(filename: str, csv_name: str = None):
    """Reads evap file into a pandas dataframe and optionally writes the data as a csv

    Parameters
    ----------
    file_name : str
        path to .EVA file
    csv_name : str, optional
        path to csv file to write to, by default None

    Returns
    -------
    DataFrame
        Evaporation data
    """
    with open(filename, "rt") as file:
        evap_lines = file.readlines()
    data = []
    for line in evap_lines:
        if line[0] != "*":
            data.append(line.split())
    flo_df = pd.DataFrame(data)
    flo_df = flo_df.dropna()
    flo_df = flo_df.pivot(index=0, columns=1).transpose().swaplevel().sort_index()
    flo_df = flo_df.reset_index()
    flo_df = flo_df.rename(columns={1: "year", "level_1": "month"})
    flo_df["month"] = flo_df["month"] - 1
    flo_df["year"] = flo_df.year.astype(int)
    flo_df.insert(
        0,
        "date",
        flo_df.apply(lambda row: np.datetime64(f"{row.year}-{row.month:02d}"), axis=1),
    )
    flo_df = flo_df.drop(columns=["year", "month"])
    flo_df = flo_df.set_index("date")
    flo_df = flo_df.astype(float)
    
    if csv_name:
        flo_df.write_csv(csv_name)
        
    return flo_df


def dat_to_df(wrap_file_path, csv_name: str = None):
    """TRAVIS THURBER
    Converts a WRAP input file (.DAT extension) to a CSV file containing water rights records.
    Other records are currently ignored.

    :param wrap_file_path: path to the WRAP .DAT file

    :return: None
    """

    # read the lines from the file
    file_path = Path(wrap_file_path)
    f = open(file_path, 'r')
    lines = f.readlines()

    # create lists for storing each type of data
    water_rights = []

    # loop through each line and only parse lines that start with WR
    for line in lines:

        if not (line.startswith('WR')):
            continue

        #if (line.startswith(('WRA-ZERO','WRENVCAP','WRDRTNUM','WRDRTCON','WRDRTKEY'))):
        #    continue

        # current position in line
        spot = 0

        # dictionary for the data in this line
        datum = {}

        # loop through each column and parse reservoir data from the line
        for col in COL_WATER_RIGHT:
            try:
                value = line[spot:spot + col['length']].strip()
                if (len(value) > 0) and (value == len(value) * '*'):
                    value = col['dtype'](np.nan)
                    print(f"WARNING: Value overflow for water right for column {col['name']}:")
                    print(f"    {line}")
                elif (len(value) == 0):
                    if (col['dtype'] == np.int16):
                        value = 0
                    elif (col['dtype'] == np.float32):
                        value = col['dtype'](np.nan)
                    else:
                        value = col['dtype'](value)
                else:
                    value = col['dtype'](value)
                datum[col['name']] = value
                spot += col['length']
            except Exception as e:
                print("")
                print(f"Error on line {line}")
                print(f"Column {col['name']}")
                print("")
                raise(e)
        water_rights.append(datum)

    # create data frames from each type of data
    water_rights = pd.DataFrame(water_rights)

    # creat csv file
    if csv_name:
        water_rights.to_csv(csv_name, index=False)
    return water_rights

# @profile
def out_to_csvs(out_file, csv_folder, csvs_to_write=None):
    """TRAVIS THURBER
    Converts a WRAP output file (.OUT extension) to four CSV or parquet files, one for each type of data

    :param wrap_file_path: path to the WRAP .OUT file

    :return: None
    """
    csv_types = ["diversions", "flow_rights", "control_points", "reservoirs"]
    if csvs_to_write is None:
        csvs_to_write = csv_types
            
    if "diversions" in csvs_to_write:
        write_diversions = True
    else:
        write_diversions = False

    if "flow_rights" in csvs_to_write:
        write_flow_rights = True
    else:
        write_flow_rights = False
        
    if "control_points" in csvs_to_write:
        write_control_points = True
    else:
        write_control_points = False
        
    if "reservoirs" in csvs_to_write:
        write_reservoirs = True
    else:
        write_reservoirs = False

            

    # read the lines from the file
    file_path = Path(out_file)
    print(f"Parsing WRAP file {file_path.name}...")
    f = open(file_path, 'r')
    lines = f.readlines()

    # read and parse the meta data line
    meta = lines[N_HEADER - 1].split()
    start_year = int(meta[0])
    n_years = int(meta[1])
    n_water_rights = int(meta[3])
    n_control_points = int(meta[2])
    n_reservoirs = int(meta[4])

    # create lists for storing each type of data
    data_diversions = []
    data_flow_rights = []
    data_control_points = []
    data_reservoirs = []

    # loop through each year and month of data
    for i_year in np.arange(n_years):
        for i_month in np.arange(12):
            n_month = i_year * 12 + i_month

            # loop through each line of diversion/flow right data, and split into column
            for line in lines[
                N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) :
                N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights
            ]:

                # current position in line
                spot = 0

                # dictionary for the data in this line
                datum = {}

                # determine if this line is a flow right or a diversion
                is_flow_right = line.startswith('IF')
                # add the year if flow right
                if is_flow_right:
                    datum['year'] = np.int16(start_year + i_year)

                # loop through each column and parse diversion or flow right data from the line
                for col in (COL_FLOW_RIGHTS if is_flow_right else COL_DIVERSION_RIGHTS):
                    if col['name'] != 'IF':
                        value = line[spot:spot + col['length']].strip()
                        if (len(value) > 0) and (value == len(value) * '*'):
                            value = col['dtype'](np.nan)
                            # print(f"WARNING: Value overflow for {'flow_right' if is_flow_right else 'diversion'} for column {col['name']} in year {start_year + i_year} month {i_month + 1}:")
                            # print(f"    {line}")
                        else:
                            value = col['dtype'](value)
                        datum[col['name']] = value
                    spot += col['length']
                if is_flow_right:
                    data_flow_rights.append(datum)
                else:
                    data_diversions.append(datum)

            # loop through each line of control point data, and split into column
            for line in lines[
                N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights :
                N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights + n_control_points
            ]:

                # current position in line
                spot = 0

                # dictionary for the data in this line
                datum = {}

                # add the year and month
                datum['year'] = np.int16(start_year + i_year)
                datum['month'] = np.int16(i_month + 1)

                # loop through each column and parse control point data from the line
                for col in COL_CONTROL_POINTS:
                    value = line[spot:spot + col['length']].strip()
                    if (len(value) > 0) and (value == len(value) * '*'):
                        value = col['dtype'](np.nan)
                        # print(f"WARNING: Value overflow for control_point for column {col['name']} in year {start_year + i_year} month {i_month + 1}:")
                        # print(f"    {line}")
                    else:
                        value = col['dtype'](value)
                    datum[col['name']] = value
                    spot += col['length']
                data_control_points.append(datum)

            # loop through each line of reservoir data, and split into column
            for line in lines[
                N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights + n_control_points :
                N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights + n_control_points + n_reservoirs
            ]:

                # current position in line
                spot = 0

                # dictionary for the data in this line
                datum = {}

                # add the year and month
                datum['year'] = np.int16(start_year + i_year)
                datum['month'] = np.int16(i_month + 1)

                # loop through each column and parse reservoir data from the line
                for col in COL_RESERVOIR:
                    value = line[spot:spot + col['length']].strip()
                    if (len(value) > 0) and (value == len(value) * '*'):
                        value = col['dtype'](np.nan)
                        # print(f"WARNING: Value overflow for reservoir for column {col['name']} in year {start_year + i_year} month {i_month + 1}:")
                        # print(f"    {line}")
                    else:
                        value = col['dtype'](value)
                    datum[col['name']] = value
                    spot += col['length']
                data_reservoirs.append(datum)

    # create data frames from each type of data
    if write_diversions:
        data_diversions = pd.DataFrame(data_diversions)
        data_diversions.to_csv(join(csv_folder, f'{file_path.stem}_diversions.csv'), index=False)
        print(f'{file_path.stem}_diversions.csv')
    
    if write_flow_rights:
        data_flow_rights = pd.DataFrame(data_flow_rights)
        data_flow_rights.to_csv(join(csv_folder, f'{file_path.stem}_flow_rights.csv'), index=False)
        print(f'{file_path.stem}_flow_rights.csv')
    
    if write_control_points:
        data_control_points = pd.DataFrame(data_control_points)
        data_control_points.to_csv(join(csv_folder, f'{file_path.stem}_control_points.csv'), index=False)
        print(f'{file_path.stem}_control_points.csv')
    
    if write_reservoirs:
        data_reservoirs = pd.DataFrame(data_reservoirs)
        data_reservoirs.to_csv(join(csv_folder, f'{file_path.stem}_reservoirs.csv'), index=False)
        print(f'{file_path.stem}_reservoirs.csv')


def process_right_sectors(dat_file_path, filter_sectors=True, sectors=None):
    dat = pd.read_csv(dat_file_path)
    if filter_sectors:
        if sectors is None:
            sectors = ["IND", "IRR", "MIN", "MUN", "POW", "REC"]

        def process_use(row):
            for sector in sectors:
                try:
                    if sector in row.use:
                        return sector
                except TypeError:
                    return "nan"

        dat.use = dat.apply(process_use, axis=1)
        dat = dat[dat.use.isin(sectors)]
    
    return dat
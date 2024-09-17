import pandas as pd
from pandas import DataFrame

def process_diversion_csv(diversions: DataFrame):
    diversions["date"] = pd.to_datetime(diversions[["year", "month"]].assign(DAY=1))
    shortage = diversions.diversion_or_energy_shortage 
    target = diversions.diversion_or_energy_target
    diversions["shortage_ratio"] = 1 - ((target - shortage) / target)
    ratios = diversions.pivot_table(
        index="date",
        columns="water_right_identifier",
        values="shortage_ratio",
        dropna=False
    )
    return ratios
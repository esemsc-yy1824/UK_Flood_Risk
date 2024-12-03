import os
import pandas as pd

"""
Functions for mapping and data processing.
"""

def get_house_head(postcode):
    """
    Retrieve household and headcount data for a given postcode sector.

    Parameters
    ----------
    postcode : str
        Full postcode string (e.g., "AB12 3CD"). The function extracts the
        sector (e.g., "AB12 3") for matching.

    Returns
    -------
    tuple
        - First element: Household count (int).
        - Second element: Headcount (int).
    """
    
    current_path = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(current_path, '../resources_data/sector_data.csv'))

    postcode_first = postcode.split()[0]  # first part
    postcode_second = postcode.split()[1][0]  # first character of the second part
    postcodeSector = postcode_first + " " + postcode_second

    # Filter for matching postcode sector
    matching_data = df[df["postcodeSector"] == postcodeSector]
    if matching_data.empty:
        return None, None

    # Extract scalar values for households and headcount
    households = int(matching_data["households"].iloc[0])
    headcount = int(matching_data["headcount"].iloc[0])
    return households, headcount

def normalise_rainfall(df):
    """
    Normalize rainfall values to millimeters (mm) if the unit is not already in mm.

    If the `unitName` is not "mm", the corresponding `value` is multiplied by 1000 
    to convert it to millimeters, and the `unitName` is updated to "mm".

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing rainfall data with columns `unitName` and `value`.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with normalized rainfall values in millimeters.

    Examples
    --------
    >>> df = pd.DataFrame({'unitName': ['m', 'mm'], 'value': [0.5, 10]})
    >>> normalise_rainfall(df)
    """
    df.loc[df["unitName"] != "mm", "value"] *= 1000
    df.loc[df["unitName"] != "mm", "unitName"] = "mm"
    return df


def Rainfall_merged_data(station_name):
    """ 
    Merge station data with typical and wet day rainfall data for the given station.

    This function reads the station, typical day, and wet day rainfall data, filters 
    for the specified station, normalizes the rainfall values, and merges the data 
    into a single DataFrame.

    Parameters
    ----------
    station_name : str
        The name of the station to retrieve and merge data for.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the merged station and rainfall data, with columns for
        typical and wet day rainfall values.

    Notes
    -----
    - The rainfall values are normalized to millimeters if they are not already.
    """
    # read the station data
    current_path = os.path.dirname(__file__)
    station_df = pd.read_csv(os.path.join(current_path, '../resources_data/stations.csv'))
    station_info = station_df[station_df["stationName"] == station_name][['stationReference', 'latitude', 'longitude']]
    
    if station_info.empty:
        print(f"No station found with name: {station_name}")
        return None
    
    # read the rainfall data
    typical_data = pd.read_csv(os.path.join(current_path, "../resources_data/typical_day.csv"))
    wet_data = pd.read_csv(os.path.join(current_path, "../resources_data/wet_day.csv"))
    typical_data['value'] = pd.to_numeric(typical_data['value'], errors='coerce')
    wet_data['value'] = pd.to_numeric(wet_data['value'], errors='coerce')

    # normalise the rainfall data
    typical_data = normalise_rainfall(typical_data)
    wet_data = normalise_rainfall(wet_data)

    # merge the station data with the rainfall data
    merged_data = pd.merge(
        station_info,
        typical_data[["stationReference", "parameter", "value"]],
        on="stationReference",
        how="left"
    ).rename(columns={"parameter": "typical_parameter", "value": "typical_value"})

    merged_data = pd.merge(
        merged_data,
        wet_data[["stationReference", "parameter", "value"]],
        on="stationReference",
        how="left"
    ).rename(columns={"parameter": "wet_parameter", "value": "wet_value"})

    merged_data.drop_duplicates(inplace=True)
    merged_data.dropna(inplace=True)

    return merged_data
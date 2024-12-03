import pandas as pd
import plotly.express as px
from .mapping_tools import get_house_head, Rainfall_merged_data

def Risk_HeatMap(file_path):
    """
    Generate a heatmap of risk levels using Plotly's density map.

    This function creates a geographic heatmap that visualizes risk levels 
    based on latitude, longitude, and a risk label. The data is loaded from 
    a CSV file, and the map uses a custom color scale to indicate varying 
    levels of risk.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the risk data. The file must include 
        the following columns:
        - 'Latitude': Latitude of the location (float).
        - 'Longitude': Longitude of the location (float).
        - 'riskLabel': Risk level for the location (numeric, from 1 to 7).
        - 'postcode' (optional): Postcode of the location (string).

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object containing the heatmap visualization.

    Raises
    ------
    ValueError
        If missing values are detected in the required columns 
        ('Latitude', 'Longitude', 'riskLabel').

    Notes
    -----
    - The `riskLabel` column is expected to be numeric. Non-numeric values 
      will be coerced into NaN and can cause missing value errors.
    - The heatmap uses a custom color scale ranging from blue (low risk) to red (high risk).

    Examples
    --------
    >>> file_path = "risk_data.csv"
    >>> fig = Risk_HeatMap(file_path)
    >>> fig.show()
    """

    df = pd.read_csv(file_path)
    df['riskLabel'] = pd.to_numeric(df['riskLabel'], errors='coerce')

    if df[['Latitude', 'Longitude', 'riskLabel']].isnull().any().any():
        raise ValueError("Missing values detected in required columns.")

    fig = px.density_mapbox(
        df,
        lat='Latitude',
        lon='Longitude',
        z='riskLabel',
        radius=15,
        center=dict(lat=51.4254, lon=-0.2137),
        zoom=8,
        mapbox_style="open-street-map",
        title="Risk Heatmap",
        color_continuous_scale=["blue", "cyan", "yellow", "orange", "red"],  # Custom color scale
        opacity=0.6
    )

    return fig

def add_high_risk_markers(fig, df):
    """
    Add high-risk markers to the Plotly map for areas with `riskLabel` equal to 7.

    This function identifies high-risk locations from the DataFrame, retrieves 
    additional information (households and headcount) for these locations, and 
    adds markers to an existing Plotly map figure. The markers are displayed in red 
    and include additional data in hover tooltips.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The existing Plotly map figure to which the markers will be added.
    df : pandas.DataFrame
        The DataFrame containing map data. It must include the following columns:
        - `Latitude`: Latitude of the location (float).
        - `Longitude`: Longitude of the location (float).
        - `postcode`: Postcode of the location (string).
        - `riskLabel`: Risk level for the location (numeric).

    Returns
    -------
    plotly.graph_objects.Figure
        Updated Plotly figure with added high-risk markers.

    Notes
    -----
    - The `riskLabel` column is used to filter for locations with a value of 7.
    - Additional data for hover tooltips (`households` and `headcount`) are 
      retrieved using the `get_house_head` function.

    Raises
    ------
    KeyError
        If the required columns are missing from the input DataFrame.

    Examples
    --------
    >>> fig = px.density_mapbox(...)  # Existing map figure
    >>> df = pd.DataFrame({
    ...     'Latitude': [51.5, 51.6],
    ...     'Longitude': [-0.1, -0.2],
    ...     'postcode': ['SW1A', 'SW1B'],
    ...     'riskLabel': [7, 6]
    ... })
    >>> updated_fig = add_high_risk_markers(fig, df)
    >>> updated_fig.show()
    """
    high_risk_df = df[df['riskLabel'] == 7]

    # Retrieve households and headcount as scalars
    households, headcount = zip(*high_risk_df['postcode'].apply(get_house_head))
    high_risk_df['households'] = households
    high_risk_df['headcount'] = headcount

    # Handle missing values
    high_risk_df['households'] = high_risk_df['households'].fillna("Unknown")
    high_risk_df['headcount'] = high_risk_df['headcount'].fillna("Unknown")

    # Add a scatter layer for markers
    scatter = px.scatter_mapbox(
        high_risk_df,
        lat='Latitude',
        lon='Longitude',
        hover_name='postcode',
        hover_data={
            'households': True,
            'headcount': True,
        },
        color_discrete_sequence=["red"],
        title="High-Risk Areas",
    )
    fig.add_trace(scatter.data[0])  # Add scatter trace to the existing figure
    return fig

def Risk_HeatMap_with_markers(file_path):
    """
    Generate a risk heatmap with high-risk markers using Plotly.

    This function creates a density heatmap to visualize risk levels across locations
    and adds specific markers for high-risk areas (`riskLabel` equal to 7) to the map.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the risk data. The file must include the following columns:
        - `Latitude`: Latitude of the location (float).
        - `Longitude`: Longitude of the location (float).
        - `riskLabel`: Risk level for the location (numeric, from 1 to 7).
        - `postcode` (optional): Postcode of the location (string).

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object containing the combined heatmap and high-risk markers.

    Notes
    -----
    - The heatmap is generated using the `Risk_HeatMap` function.
    - Markers for high-risk areas (with `riskLabel` == 7) are added using the 
      `add_high_risk_markers` function.

    Raises
    ------
    ValueError
        If missing values are detected in required columns when generating the heatmap.
    KeyError
        If the required columns for high-risk markers are missing in the data.

    Examples
    --------
    >>> file_path = "risk_data.csv"
    >>> fig = Risk_HeatMap_with_markers(file_path)
    >>> fig.show()
    """
    df = pd.read_csv(file_path)
    fig = Risk_HeatMap(file_path)
    fig = add_high_risk_markers(fig, df)
    return fig

def Wet_Rainfall_HeatMap(station_name):
    """
    Create a heatmap with rainfall data on wet days for a specified station.

    This function visualizes rainfall intensity on wet days using a density heatmap.
    It processes data from the specified station, filters for rainfall-related values,
    and generates a Plotly map to display the data geographically.

    Parameters
    ----------
    station_name : str
        The name of the weather station to filter rainfall data.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object containing the heatmap visualization.

    Notes
    -----
    - The rainfall data is obtained by calling `Rainfall_merged_data`, which
      provides the required data for the specified station.
    - The map uses a custom color scale ranging from blue (low rainfall) to red (high rainfall).
    - The center of the map is dynamically set based on the mean latitude and longitude
      of the rainfall data.

    Raises
    ------
    KeyError
        If required columns (`latitude`, `longitude`, `wet_value`, `wet_parameter`) 
        are missing in the merged data.

    Examples
    --------
    >>> station_name = "Example Station"
    >>> fig = Wet_Rainfall_HeatMap(station_name)
    >>> fig.show()
    """
    merged_data = Rainfall_merged_data(station_name)

    # Filter data for rainfall
    rainfall_data = [
        [row['latitude'], row['longitude'], row['wet_value']]
        for _, row in merged_data.iterrows() if row['wet_parameter'] == 'rainfall'
    ]
    
    rainfall_df = pd.DataFrame(rainfall_data, columns=['Latitude', 'Longitude', 'Rainfall'])

    fig = px.density_mapbox(
        rainfall_df,
        lat='Latitude',
        lon='Longitude',
        z='Rainfall',
        radius=15,
        center=dict(lat=merged_data['latitude'].mean(), lon=merged_data['longitude'].mean()),
        zoom=8,
        mapbox_style="open-street-map",
        title="Wet Day Rainfall Heatmap",
        color_continuous_scale=["blue", "cyan", "yellow", "red"],
        opacity=0.6  # Set opacity for transparency
    )
    
    # Customize the color scale
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Rainfall (mm)",
            tickvals=[-2150, -975, 200],  # Custom tick values for the color bar
            ticktext=["Low", "Medium", "High"]
        )
    )

    return fig

def Typical_Rainfall_HeatMap(station_name):
    """
    Create a heatmap with typical day rainfall data for a specified station.

    This function visualizes typical day rainfall values using a density heatmap.
    It processes the rainfall data for the given station, filters for typical day 
    rainfall, and generates a Plotly map to display the data geographically.

    Parameters
    ----------
    station_name : str
        The name of the weather station to filter typical day rainfall data.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object containing the heatmap visualization.

    Notes
    -----
    - The typical day rainfall data is obtained by calling `Rainfall_merged_data`, 
      which returns the required data for the specified station.
    - The map uses a custom color scale, ranging from blue (low rainfall) to red (high rainfall).
    - The center of the map is dynamically set based on the mean latitude and longitude
      of the rainfall data.
    - The color scale is customized with a color bar, which can be adjusted depending 
      on the station-specific data.

    Raises
    ------
    KeyError
        If required columns (`latitude`, `longitude`, `typical_value`, `typical_parameter`) 
        are missing in the merged data.

    Examples
    --------
    >>> station_name = "Example Station"
    >>> fig = Typical_Rainfall_HeatMap(station_name)
    >>> fig.show()
    """
    merged_data = Rainfall_merged_data(station_name)

    rainfall_data = [
        [row['latitude'], row['longitude'], row['typical_value']]
        for _, row in merged_data.iterrows() if row['typical_parameter'] == 'rainfall'
    ]
    
    # Prepare dataframe for Plotly
    rainfall_df = pd.DataFrame(rainfall_data, columns=['Latitude', 'Longitude', 'Rainfall'])

    # Generate the density map with Plotly
    fig = px.density_mapbox(
        rainfall_df,
        lat='Latitude',
        lon='Longitude',
        z='Rainfall',
        radius=15,
        center=dict(lat=merged_data['latitude'].mean(), lon=merged_data['longitude'].mean()),
        zoom=7,
        mapbox_style="open-street-map",
        title="Typical Day Rainfall Heatmap",
        color_continuous_scale=["blue", "cyan", "yellow", "red"],  # Color scale for rainfall
        opacity=0.6  # Set opacity for transparency
    )

    # Customize the color scale with a color bar
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Rainfall (mm)",
            tickvals=[-2150, -975, 200],  # TODO: the values shuold change while choose different station name.
            ticktext=["Low", "Medium", "High"]
        )
    )

    return fig
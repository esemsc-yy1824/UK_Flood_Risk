import pandas as pd
import plotly.express as px
from .mapping_tools import get_house_head, Rainfall_merged_data

def Risk_HeatMap(file_path):
    """
    Generate a heatmap of risk levels using Plotly's density map.

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
        Plotly figure object with the heatmap.
    """
    # Load the data
    df = pd.read_csv(file_path)
    df['riskLabel'] = pd.to_numeric(df['riskLabel'], errors='coerce')

    # Check for missing values
    if df[['Latitude', 'Longitude', 'riskLabel']].isnull().any().any():
        raise ValueError("Missing values detected in required columns.")

    # Generate the density map
    # Generate the density map with a custom color scale
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
    Add high-risk markers to the Plotly map for areas with riskLabel == 7.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to add the markers.
    df : pandas.DataFrame
        Dataframe containing the map data.

    Returns
    -------
    plotly.graph_objects.Figure
        Updated Plotly figure with high-risk markers.
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
    df = pd.read_csv(file_path)
    fig = Risk_HeatMap(file_path)  # Generate the heatmap
    fig = add_high_risk_markers(fig, df)  # Add high-risk markers
    return fig


def Wet_Rainfall_HeatMap(station_name):
    """
    Create a heatmap with rainfall data in wet day based on the given station.
    Uses Plotly for the heatmap visualization.
    """
    merged_data = Rainfall_merged_data(station_name)

    # Filter data for rainfall
    rainfall_data = [
        [row['latitude'], row['longitude'], row['wet_value']]
        for _, row in merged_data.iterrows() if row['wet_parameter'] == 'rainfall'
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
    Create a heatmap with typical day rainfall data based on the given station.
    This version uses Plotly for the heatmap visualization.
    """
    # Get the merged rainfall data
    merged_data = Rainfall_merged_data(station_name)

    # Filter data for typical rainfall
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
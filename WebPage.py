import streamlit as st
import os
from tools.mapping import Risk_HeatMap_with_markers, Wet_Rainfall_HeatMap, Typical_Rainfall_HeatMap
# from tools.mapping import Risk_HeatMap_with_markers, Wet_Rainfall_HeatMap

# Streamlit app setup
st.title("Interactive Map Generator")
st.sidebar.header("Map Options")

# Dropdown menu for function selection
map_type = st.sidebar.selectbox(
    "Select the type of map to generate:",
    ("Risk HeatMap", "Wet Rainfall HeatMap", "Typical Rainfall HeatMap")
)

if map_type == "Wet Rainfall HeatMap" or map_type == "Typical Rainfall HeatMap":
    station_name = st.sidebar.text_input("Enter Station Name:", "")    # Input for station name

# Generate the map based on user selection
if st.sidebar.button("Generate Map"):
    try:
        if map_type == "Risk HeatMap":
            current_path = os.path.dirname(__file__)  # Get the current file's directory
            forecast_path = os.path.join(current_path, 'output/postcodes_unlabelled_predicted.csv')
            fig = Risk_HeatMap_with_markers(forecast_path)
            # Set the size of the map figure
            fig.update_layout(
                autosize=False,  # Allow the figure to adjust to container size
                width=1500,      # Set width (in pixels)
                height=600      # Set height (in pixels)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif map_type == "Wet Rainfall HeatMap" and station_name:
            fig = Wet_Rainfall_HeatMap(station_name)
            fig.update_layout(
                autosize=False,  # Allow the figure to adjust to container size
                width=1500,      # Set width (in pixels)
                height=600      # Set height (in pixels)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif map_type == "Typical Rainfall HeatMap" and station_name:
            fig = Typical_Rainfall_HeatMap(station_name)
            fig.update_layout(
                autosize=False,  # Allow the figure to adjust to container size
                width=1500,      # Set width (in pixels)
                height=600      # Set height (in pixels)
            )
            st.plotly_chart(fig, use_container_width=True)
        #     st.session_state["map_object"] = Typical_Rainfall_HeatMap(station_name)
        
        else:
            st.warning("Please enter a valid station name.")
    
    except Exception as e:
        st.error(f"Error generating map: {e}")
        st.session_state["map_object"] = None



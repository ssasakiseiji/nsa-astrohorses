import streamlit as st
import pandas as pd
from exoplanet_utils import load_exoplanet_data, filter_exoplanets_by_mass, get_exoplanet_statistics

def main():
    st.title("Exoplanet Visualizer")
    
    st.sidebar.header("Settings")
    st.sidebar.markdown("Adjust the parameters to visualize the exoplanet data.")

    # Load the exoplanet data
    data = load_exoplanet_data("data/exoplanets.csv")
    
    if data is not None:
        st.write("### Exoplanet Data Overview")
        st.dataframe(data)

        # Filter by mass
        mass_threshold = st.sidebar.slider("Mass Threshold (Jupiter Masses)", min_value=0.0, max_value=10.0, value=5.0)
        filtered_data = filter_exoplanets_by_mass(data, mass_threshold)
        st.write(f"### Exoplanets with Mass < {mass_threshold} Jupiter Masses")
        st.dataframe(filtered_data)

        # Display statistics
        stats = get_exoplanet_statistics(data)
        st.write("### Exoplanet Statistics")
        st.json(stats)

        # Add more interactive visualizations here
        st.write("### Visualizations")
        feature = st.sidebar.selectbox("Select a feature to visualize", data.columns)
        st.bar_chart(data[feature].value_counts())
    else:
        st.error("Error loading data. Please check the file path.")

if __name__ == "__main__":
    main()
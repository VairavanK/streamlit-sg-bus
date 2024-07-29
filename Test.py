import streamlit as st
import requests
import pandas as pd
import numpy as np
from rtree import index
import datetime
import os
from dotenv import load_dotenv
# Hardcoded API key
load_dotenv('env.txt')
LTA_API_KEY = os.getenv('LTA_API_KEY')



# Assuming your dataframe is called 'df' with columns 'lat', 'lon', and 'description'
pd.options.mode.chained_assignment = None
df = pd.read_csv('bus_stops.csv')

# Create a spatial index
idx = index.Index()
df['spatial_index'] = range(len(df))  # New column for spatial indexing

for i, row in df.iterrows():
    idx.insert(row['spatial_index'], (row['Longitude'], row['Latitude'], row['Longitude'], row['Latitude']))
# Function to calculate minutes from now
def minutes_from_now(timestamp):
    if not timestamp:
        return "Invalid timestamp"
    try:
        current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))  # Assuming the time zone is +8
        estimated_time = datetime.datetime.fromisoformat(timestamp)
        delta = estimated_time - current_time
        return int(delta.total_seconds() // 60)
    except ValueError:
        return "Invalid timestamp"

# Function to fetch bus arrival data
def fetch_bus_arrival_data(bus_stop_code):
    # API details
    url = "http://datamall2.mytransport.sg/ltaodataservice/BusArrivalv2"
    params = {
        "BusStopCode": bus_stop_code
    }
    headers = {
        "AccountKey": LTA_API_KEY
    }

    try:
        # Get data from API
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()

        # Check if 'Services' key exists in the response
        if 'Services' not in data:
            print("Error: No services data found in the response")
            return None
        else:
            # Process data
            services = data['Services']
            rows = []

            for service in services:
                service_no = service['ServiceNo']
                next_bus1 = service.get('NextBus', {})
                next_bus2 = service.get('NextBus2', {})
                next_bus3 = service.get('NextBus3', {})

                row = {
                    "Service Number": service_no,
                    "Next Bus 1": f"{minutes_from_now(next_bus1.get('EstimatedArrival', ''))} mins, {next_bus1.get('Load', 'N/A')}, {next_bus1.get('Feature', 'N/A')}, {next_bus1.get('Type', 'N/A')}",
                    "Next Bus 2": f"{minutes_from_now(next_bus2.get('EstimatedArrival', ''))} mins, {next_bus2.get('Load', 'N/A')}, {next_bus2.get('Feature', 'N/A')}, {next_bus2.get('Type', 'N/A')}",
                    "Next Bus 3": f"{minutes_from_now(next_bus3.get('EstimatedArrival', ''))} mins, {next_bus3.get('Load', 'N/A')}, {next_bus3.get('Feature', 'N/A')}, {next_bus3.get('Type', 'N/A')}"
                }
                rows.append(row)

            # Create DataFrame
            df = pd.DataFrame(rows)
            return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

    except ValueError as e:
        print(f"Error processing JSON data: {e}")
        return None

def query_onemap(address):
    base_url = "https://www.onemap.gov.sg/api/common/elastic/search"
    params = {
        "searchVal": address,
        "returnGeom": "Y",
        "getAddrDetails": "Y"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["found"] > 0:
            result = data["results"][0]
            return {
                "address": result["ADDRESS"],
                "latitude": float(result["LATITUDE"]),
                "longitude": float(result["LONGITUDE"])
            }
        else:
            return None
    except requests.RequestException as e:
        return None


def haversine_vector(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c * 1000  # Distance in meters


# Function to get nearby descriptions
def get_nearby_descriptions(target_lat, target_lon, max_distance=500):
    # Query the spatial index (with a slightly larger bounding box)
    nearby_indices = list(idx.intersection((target_lon - 0.01, target_lat - 0.01,
                                            target_lon + 0.01, target_lat + 0.01)))

    if not nearby_indices:
        return []

    nearby_points = df[df['spatial_index'].isin(nearby_indices)]

    # Calculate distances
    distances = haversine_vector(target_lat, target_lon,
                                 nearby_points['Latitude'], nearby_points['Longitude'])

    # Filter points within max_distance
    within_distance = nearby_points[distances <= max_distance]
    within_distance['distance'] = distances[distances <= max_distance]
    within_distance['distance'] = round(within_distance['distance'], 0)
    # Sort by distance
    within_distance = within_distance.sort_values('distance')
    return within_distance[['BusStopCode', 'Description', 'distance']].reset_index(drop=True)


def main():
    st.title("Address Search App")

    # Initialize session state
    if 'selected' not in st.session_state:
        st.session_state.selected = None
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False

    # Create a form for the search input
    with st.form(key='search_form'):
        col1, col2 = st.columns([3, 1])
        with col1:
            address = st.text_input("Address Input")
        with col2:
            search_button = st.form_submit_button("Search")

    if search_button:
        st.session_state.selected = None
    if search_button or (st.session_state.search_performed and address):
        st.session_state.search_performed = True
        result = query_onemap(address)
        if result is not None:
            nearby_descriptions = get_nearby_descriptions(result['latitude'], result['longitude'])
            if not nearby_descriptions.empty:
                st.write("Search Results:")
                if st.session_state.selected is None:
                    for i, row in nearby_descriptions.iterrows():
                        if st.button(f"{row['Description']} ({row['distance']} meters)"):
                            st.session_state.selected = row
                else:
                    st.write(f"You selected: Bus Stop Code {st.session_state.selected['BusStopCode']}, Description: {st.session_state.selected['Description']}, Distance: {st.session_state.selected['distance']} meters")
                    st.dataframe(fetch_bus_arrival_data(st.session_state.selected['BusStopCode']))
            else:
                st.write("No nearby bus stops found.")
        else:
            st.error("No results found for the provided address.")

    if st.session_state.selected is not None:

        if st.button("Back"):
            st.session_state.selected = None
            st.experimental_rerun()


if __name__ == "__main__":
    main()

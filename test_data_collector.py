import requests
import json

# Function to fetch recent earthquake data from USGS

def fetch_earthquake_data():
    url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
    params = {
        'format': 'geojson',
        'limit': 10,
        'orderby': 'time'
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to display basic statistics

def display_statistics(data):
    features = data['features']
    print(f'Total Earthquakes: {len(features)}')
    for feature in features:
        magnitude = feature['properties']['mag']
        place = feature['properties']['place']
        print(f'Magnitude: {magnitude}, Location: {place}')

if __name__ == '__main__':
    earthquake_data = fetch_earthquake_data()
    display_statistics(earthquake_data)
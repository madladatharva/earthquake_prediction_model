import requests
import json

class USGSDataCollector:
    def __init__(self, endpoint='https://earthquake.usgs.gov/fdsnws/event/1/query'):
        self.endpoint = endpoint

    def fetch_data(self, start_time, end_time, min_magnitude=5.0):
        params = {
            'format': 'geojson',
            'starttime': start_time,
            'endtime': end_time,
            'minmagnitude': min_magnitude
        }
        response = requests.get(self.endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f'Error fetching data: {response.status_code}')

# Example usage:
# collector = USGSDataCollector()
# data = collector.fetch_data('2023-01-01', '2023-12-31')
# print(json.dumps(data, indent=2))
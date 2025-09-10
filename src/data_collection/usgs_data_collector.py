"""
Basic USGS data collector for earthquake data.
"""
import requests
import json
import logging

class USGSDataCollector:
    """Basic USGS earthquake data collector."""
    
    def __init__(self, endpoint='https://earthquake.usgs.gov/fdsnws/event/1/query'):
        self.endpoint = endpoint
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, start_time, end_time, min_magnitude=5.0):
        """
        Fetch earthquake data from USGS API.
        
        Args:
            start_time: Start time string (ISO format)
            end_time: End time string (ISO format)
            min_magnitude: Minimum magnitude threshold
            
        Returns:
            GeoJSON response from USGS API
            
        Raises:
            Exception: If API request fails
        """
        params = {
            'format': 'geojson',
            'starttime': start_time,
            'endtime': end_time,
            'minmagnitude': min_magnitude
        }
        
        try:
            response = requests.get(self.endpoint, params=params, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.Timeout:
            raise Exception(f'Request timeout when fetching data from {self.endpoint}')
        except requests.exceptions.ConnectionError:
            raise Exception(f'Connection error when fetching data from {self.endpoint}')
        except requests.exceptions.HTTPError:
            raise Exception(f'HTTP error {response.status_code} when fetching data')
        except requests.exceptions.RequestException as e:
            raise Exception(f'Request error: {e}')
        except json.JSONDecodeError:
            raise Exception(f'Invalid JSON response from USGS API')

# Example usage:
# collector = USGSDataCollector()
# data = collector.fetch_data('2023-01-01', '2023-12-31')
# print(json.dumps(data, indent=2))
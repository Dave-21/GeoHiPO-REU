from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from functools import lru_cache
import time


class GeoCoder:
    def __init__(self, user_agent: str = "geo_hipo_geocoder", max_retries: int = 3, delay: float = 1.0):
        self.geolocator = Nominatim(user_agent=user_agent, timeout=10)
        self.max_retries = max_retries
        self.delay = delay

    def _retry(func):
        def wrapper(self, *args, **kwargs):
            attempts = 0
            while attempts < self.max_retries:
                try:
                    return func(self, *args, **kwargs)
                except (GeocoderTimedOut, GeocoderServiceError):
                    attempts += 1
                    time.sleep(self.delay)
            return None
        return wrapper

    @lru_cache(maxsize=1024)
    @_retry
    def get_coordinates(self, place: str):
        location = self.geolocator.geocode(place)
        if location:
            return (location.latitude, location.longitude)
        return None

    @lru_cache(maxsize=1024)
    @_retry
    def get_place(self, latitude: float, longitude: float):
        location = self.geolocator.reverse((latitude, longitude), exactly_one=True)
        if location:
            return location.address
        return None

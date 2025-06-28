from __future__ import annotations
import os
import json
from typing import Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


class Geocoder:
    """Simple geocoder with caching using OpenStreetMap Nominatim."""

    def __init__(self, cache_file: str = "geocode_cache.json", user_agent: str = "geo_visualizer") -> None:
        self.cache_file = cache_file
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}
        # Increase timeout to avoid read timeouts
        self.geolocator = Nominatim(user_agent=user_agent, timeout=10)
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)

    def save_cache(self) -> None:
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception:
            pass

    def get_coords(self, place: str) -> Tuple[Optional[float], Optional[float]]:
        if not place or not isinstance(place, str):
            return None, None
        place = place.strip()
        if place in self.cache:
            return self.cache[place]
        try:
            loc = self.geocode(place)
            if loc:
                coords = (loc.latitude, loc.longitude)
            else:
                coords = (None, None)
        except Exception as e:
            print(f"Geocoding error for '{place}': {e}")
            coords = (None, None)
        self.cache[place] = coords
        return coords

    def __del__(self):
        self.save_cache()

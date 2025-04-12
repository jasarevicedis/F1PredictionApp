import fastf1
import fastf1.plotting
from fastf1.core import Session
from typing import Optional, List
import pandas as pd

class DataExtractor:
    def __init__(self, year: int, grand_prix: str, session_type: str):
        """
        Initializes a FastF1 session.
        :param session_type: 'FP1', 'FP2', 'FP3', 'Q', 'R', 'Sprint'
        """
        self.year = year
        self.grand_prix = grand_prix
        self.session_type = session_type
        self.session: Optional[Session] = None

    def load_session(self) -> None:
        fastf1.Cache.enable_cache("data/cache") 
        self.session = fastf1.get_session(self.year, self.grand_prix, self.session_type)
        self.session.load()

    def get_laps(self) -> pd.DataFrame:
        if self.session is None:
            self.load_session()
        return self.session.laps

    def get_driver_laps(self, driver: str) -> pd.DataFrame:
        return self.get_laps().pick_driver(driver)

    def get_telemetry_by_lap(self, driver: str, lap_number: int) -> pd.DataFrame:
        laps = self.get_driver_laps(driver)
        lap = laps.loc[laps['LapNumber'] == lap_number].iloc[0]
        return lap.get_car_data().add_distance()

    def get_weather_data(self) -> pd.DataFrame:
        if self.session is None:
            self.load_session()
        return self.session.weather_data

    def get_driver_list(self) -> List[str]:
        return self.get_laps()['Driver'].unique().tolist()


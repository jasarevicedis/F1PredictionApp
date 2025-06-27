import fastf1
import fastf1.plotting
from fastf1.core import Session
from fastf1 import get_session
from typing import Optional, List
import pandas as pd
from constants import Driver, TyreCompound, Track, SessionType

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
        print(self.year, self.grand_prix, self.session_type)
        #fastf1.Cache.enable_cache("data/cache") 
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
    
    def get_session_data_by_driver(training: SessionType, driver: Driver, season: int, track: Track):
        session = get_session(season, track, training.value)
        session.load()

        driver_laps = session.laps.pick_drivers(driver.value)
        weather_data = driver_laps.get_weather_data()
        # car_data = driver_laps.get_car_data()
        # position_data = driver_laps.get_pos_data()
        '''
        driver_laps.drop(columns= [
            "Time",
            "LapNumber",
            "DriverNumber", 
            "PitOutTime", 
            "PitInTime",
            "Sector1Time",
            "SpeedI1",
            "SpeedI2",
            "Sector2Time",
            "Sector3Time",
            "Sector1SessionTime",
            "Sector2SessionTime",
            "Sector3SessionTime",
            "SpeedFL",
            "SpeedST",
            "Deleted",
            "DeletedReason",
            "Position",
            "IsPersonalBest",
            "LapStartTime",
            "LapStartDate",
        ], inplace=True)

        weather_data.drop(columns= [
            "WindSpeed",
            "WindDirection",
            "Pressure",
            "Humidity",
            "Time"
        ], inplace=True)
        '''

        # Reset indices just in case
        driver_laps.reset_index(drop=True, inplace=True)
        weather_data.reset_index(drop=True, inplace=True)
        combined_data = pd.concat([driver_laps, weather_data], axis=1)

        return combined_data
    
    def get_history_data_by_driver():
        return 
    
    

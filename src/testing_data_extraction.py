from data_fetching import DataExtractor
from constants import Driver, TyreCompound, Track, SessionType
import numpy as np
import pandas as pd
import datetime

#input
session_season = 2024
session_track = Track.MONZA
session_type = SessionType.RACE
driver = Driver.LEC



stints = []
'''
for driver in Driver:
    for session_type in SessionType:
        if session_type != SessionType.QUALIFYING and session_type != SessionType.RACE:
            laps = DataExtractor.get_session_data_by_driver(
                training=session_type,
                driver=driver,
                season = session_season,
                track = session_track
            )

            curr_stints = {stint: group.reset_index(drop=True) for stint, group in laps.groupby('Stint')}
            stints.append(curr_stints)

print(stints)
'''
#output example
'''
    Driver                LapTime  Stint Compound  TyreLife  FreshTyre     Team TrackStatus  FastF1Generated  IsAccurate  AirTemp  Rainfall  TrackTemp
0    NOR 0 days 00:01:32.952000    2.0     SOFT       1.0       True  McLaren           1            False       False     27.1     False       50.1
1    NOR 0 days 00:01:13.226000    2.0     SOFT       2.0       True  McLaren           1            False        True     27.1     False       50.3
2    NOR 0 days 00:01:46.924000    2.0     SOFT       3.0       True  McLaren           1            False        True     27.0     False       50.9
3    NOR 0 days 00:01:28.204000    2.0     SOFT       4.0       True  McLaren           1            False        True     27.1     False       50.9
4    NOR 0 days 00:01:13.684000    2.0     SOFT       5.0       True  McLaren           1            False        True     27.1     False       51.3
5    NOR 0 days 00:01:43.498000    2.0     SOFT       6.0       True  McLaren           1            False        True     27.0     False       50.4
6    NOR                    NaT    2.0     SOFT       7.0       True  McLaren          12            False       False     27.1     False       50.3
'''

# Example Leclerc data for Suzuka 2024
#input
session_season = 2024
session_track = Track.SUZUKA
session_type = SessionType.RACE
driver = Driver.LEC

extractor = DataExtractor(2024, "Suzuka", 'R')
session = extractor.load_session()
laps = DataExtractor.get_session_data_by_driver(
                training=session_type,
                driver=driver,
                season = session_season,
                track = session_track
            )

laps_lap_number = laps['LapNumber']
laps_lap_sector1 = (laps['Sector1Time'] / np.timedelta64(1,'s')).astype(float)
laps_lap_sector2 = (laps['Sector2Time'] / np.timedelta64(1,'s')).astype(float)
laps_lap_sector3 = (laps['Sector3Time'] / np.timedelta64(1,'s')).astype(float)
laps_lap_time = laps['LapTime'].dt.total_seconds()

weather = extractor.get_weather_data()
rainfall = weather['Rainfall']
rainfall = np.where(rainfall == True, 1, 0)
temperature = weather['TrackTemp']

driver_list = [driver] * len(laps_lap_number)
gp_list = [extractor.session.event['Location']] * len(laps_lap_number)

compound = laps['Compound']


list = list(zip(driver_list, gp_list, laps_lap_number, laps_lap_sector1, laps_lap_sector2, laps_lap_sector3, 
                laps_lap_time, rainfall, temperature, compound))

race_data = pd.DataFrame(list,columns= ['Driver', 'GP', 'Lap', 'S1', 'S2', 'S3', 'LapTime', 'Rain', 'TrackTemp', 'Compound'])

print(race_data)







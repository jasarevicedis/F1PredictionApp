from data_fetching import DataExtractor
from constants import Driver, TyreCompound, Track, SessionType

#input
session_season = 2022
session_track = Track.MONACO



stints = []

for driver in Driver:
    for session_type in SessionType:
        if session_type != SessionType.QUALIFYING and session_type != SessionType.RACE:
            laps = DataExtractor.get_training_data_by_driver(
                training=session_type,
                driver=driver,
                season = session_season,
                track = session_track
            )

            curr_stints = {stint: group.reset_index(drop=True) for stint, group in laps.groupby('Stint')}
            stints.append(curr_stints)

print(stints)


from data_fetching import DataExtractor
from constants import Driver, TyreCompound, Track, SessionType

laps = DataExtractor.get_training_data_by_driver(
    training=SessionType.FP1,
    driver=Driver.LEC,
    season = 2022,
    track = Track.MONACO
)

print(laps)


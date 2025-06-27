from data_fetching import DataExtractor
from constants import Driver, TyreCompound, Track, SessionType
import numpy as np
import pandas as pd
import datetime
import os 

#parameters
session_track = Track.CANADA
years_to_fetch = [2022, 2023, 2024]
output_base_dir = "Data/Data_Canada"

current_year = datetime.datetime.now().year
if current_year >= 2025:
    years_to_fetch.append(2025)

all_f1_data = {}
os.makedirs(output_base_dir, exist_ok=True)
print(f"Base output directory '{output_base_dir}' ensured.")

for year in years_to_fetch:
    session_types_to_fetch = [SessionType.FP1, SessionType.FP2, SessionType.FP3]
    # Only add Race sessions if the year is current or in the past (no future races)
    if year <= current_year:
        session_types_to_fetch.append(SessionType.RACE)

    for session_type in session_types_to_fetch:
        print(f"--- Fetching data for {year} - {session_track.value} - {session_type.value} ---")
        try:
            extractor = DataExtractor(year, session_track.value, session_type.value)
            extractor.load_session()
            weather_data = extractor.get_weather_data()
            if weather_data is None or weather_data.empty:
                print(f"  No weather data available for {year} {session_track.value} {session_type.value}. Using default values for weather.")
                session_rainfall = False
                session_track_temp = np.nan
            else:
                # Take the first available weather values for the session, assuming they are consistent.
                session_rainfall = weather_data['Rainfall'].iloc[0] if 'Rainfall' in weather_data.columns else False
                session_track_temp = weather_data['TrackTemp'].iloc[0] if 'TrackTemp' in weather_data.columns else np.nan

            # Convert boolean rainfall to a numeric (0 or 1) representation.
            session_rainfall_numeric = 1 if session_rainfall else 0

            # Define a unique identifier for the current session
            session_identifier = f"{year}_{session_type.value}"

            for driver in Driver:
                laps = DataExtractor.get_session_data_by_driver(
                    training=session_type,
                    driver=driver,
                    season=year,
                    track=session_track
                )

                if laps is None or laps.empty:
                    print(f"    No lap data found for {driver.value} in {year} {session_track.value} {session_type.value}. Skipping.")
                    continue

                # --- Data Transformation and Preparation ---
                # Extract lap number and convert timedelta objects to total seconds for sector and lap times.
                # Use .fillna(np.nan) to handle potential missing time values gracefully.
                laps_lap_number = laps['LapNumber']
                laps_lap_sector1 = laps['Sector1Time'].dt.total_seconds().fillna(np.nan)
                laps_lap_sector2 = laps['Sector2Time'].dt.total_seconds().fillna(np.nan)
                laps_lap_sector3 = laps['Sector3Time'].dt.total_seconds().fillna(np.nan)
                laps_lap_time = laps['LapTime'].dt.total_seconds().fillna(np.nan)

                compound = laps['Compound']
                if 'Team' not in laps.columns:
                    print(f"    'Team' column not found in laps data for {driver.value} in {year} {session_track.value} {session_type.value}. Skipping this driver's data for team structuring.")
                    continue
                team = laps['Team'].iloc[0]

                driver_list = [driver.value] * len(laps_lap_number)
                gp_list = [session_track.value] * len(laps_lap_number) # Grand Prix name
                year_list = [year] * len(laps_lap_number)
                session_type_list = [session_type.value] * len(laps_lap_number)
                rainfall_list = [session_rainfall_numeric] * len(laps_lap_number)
                track_temp_list = [session_track_temp] * len(laps_lap_number)

                # Create a Pandas DataFrame for the current driver's laps in this specific session.
                current_driver_session_data = pd.DataFrame({
                    'Driver': driver_list,
                    'GP': gp_list,
                    'Year': year_list,
                    'SessionType': session_type_list,
                    'Lap': laps_lap_number,
                    'S1': laps_lap_sector1,
                    'S2': laps_lap_sector2,
                    'S3': laps_lap_sector3,
                    'LapTime': laps_lap_time,
                    'Rain': rainfall_list,
                    'TrackTemp': track_temp_list,
                    'Compound': compound,
                    'Team': team
                })

                # --- Structure the Data by Team, Driver, and Session ---
                # Check if the team exists in the main dictionary, if not, add it.
                if team not in all_f1_data:
                    all_f1_data[team] = {}
                # Check if the driver exists under the team, if not, initialize an empty dictionary for their sessions.
                if driver.value not in all_f1_data[team]:
                    all_f1_data[team][driver.value] = {}

                all_f1_data[team][driver.value][session_identifier] = current_driver_session_data

                # --- Create Folders and Save to Excel ---
                team_dir = os.path.join(output_base_dir, team)
                driver_dir = os.path.join(team_dir, driver.value)

                os.makedirs(driver_dir, exist_ok=True)

                file_name = f"{session_identifier}.xlsx"
                file_path = os.path.join(driver_dir, file_name)

                current_driver_session_data.to_excel(file_path, index=False)
                print(f"    Saved data for {driver.value} in {session_identifier} to '{file_path}'")

        except Exception as e:
            print(f"An error occurred while fetching/saving data for {year} {session_track.value} {session_type.value}: {e}")
            continue

print("\n\n--- Aggregated F1 Data Summary (Separated by Session) ---")
if not all_f1_data:
    print("No data was retrieved for the specified criteria. Please check input parameters and data availability.")
else:
    for team, drivers_data in all_f1_data.items():
        print(f"\nTeam: {team}")
        for driver, sessions_data in drivers_data.items():
            print(f"  Driver: {driver}")
            if not sessions_data:
                print("    No session data available for this driver.")
                continue
            for session_id, df in sessions_data.items():
                print(f"    Session: {session_id}")
                print(f"      Total laps retrieved: {len(df)}")
                if not df.empty:
                    print(f"      First 5 rows of data:\n        {df.head()}")
                else:
                    print("      No lap data available for this session.")
                print("-" * 40) 
            print("=" * 50) 

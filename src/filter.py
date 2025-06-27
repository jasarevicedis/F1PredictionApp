import os
import pandas as pd
from constants import Track, TyreCompound, TyreClass
from track_data import TRACK_RACE_DATA

base_data_folder = "data"

tracks_to_process = [
    ("Data_Australia", Track.AUSTRALIA),
    ("Data_China", Track.CHINA),
    ("Data_Suzuka", Track.SUZUKA),
    ("Data_Bahrain", Track.BAHRAIN),
    ("Data_Jeddah", Track.JEDDAH),
    ("Data_Miami", Track.MIAMI),
    ("Data_Imola", Track.IMOLA),
    ("Data_Monaco", Track.MONACO),
    ("Data_Barcelona", Track.BARCELONA),
    ("Data_Canada", Track.CANADA),
]

print(f"--- Starting to add TyreClass column for multiple tracks ---")

for track_folder_name, current_track_enum_param in tracks_to_process:
    input_track_dir = os.path.join(base_data_folder, track_folder_name)

    print(f"\nProcessing data for track: {current_track_enum_param.value} (Folder: '{track_folder_name}')")

    if not os.path.isdir(input_track_dir):
        print(f"  Error: Track data folder '{input_track_dir}' not found. Skipping this track.")
        continue 
    else:
        # Walk through the directory structure: Team -> Driver -> Session.xlsx
        for team_name in os.listdir(input_track_dir):
            team_dir = os.path.join(input_track_dir, team_name)
            if os.path.isdir(team_dir):
                for driver_name in os.listdir(team_dir):
                    driver_dir = os.path.join(team_dir, driver_name)
                    if os.path.isdir(driver_dir):
                        for file_name in os.listdir(driver_dir):
                            if file_name.endswith(".xlsx"):
                                file_path = os.path.join(driver_dir, file_name)
                                try:
                                    # Extract year from the file name 
                                    parts = file_name.replace(".xlsx", "").split('_')
                                    if len(parts) >= 1 and parts[0].isdigit(): 
                                        current_year = int(parts[0])

                                        df = pd.read_excel(file_path)

                                        if not df.empty and 'Compound' in df.columns:
                                            current_track_enum = current_track_enum_param

                                            if current_track_enum in TRACK_RACE_DATA and current_year in TRACK_RACE_DATA[current_track_enum]:
                                                compound_allocation = TRACK_RACE_DATA[current_track_enum][current_year]["compound_allocation"]

                                                df['TyreClass'] = pd.NA

                                                # Map 'Compound' (SOFT, MEDIUM, HARD) to 'TyreClass' (C1, C2, C3, etc.)
                                                for tyre_compound, tyre_class in compound_allocation.items():
                                                    df.loc[df['Compound'] == tyre_compound.value, 'TyreClass'] = tyre_class.value
                                                
                                                # Handle INTERMEDIATE and WET tyres if they are present
                                                df.loc[df['Compound'] == TyreCompound.INTER.value, 'TyreClass'] = TyreCompound.INTER.value
                                                df.loc[df['Compound'] == TyreCompound.WET.value, 'TyreClass'] = TyreCompound.WET.value

                                                df.to_excel(file_path, index=False)
                                                print(f"    Updated '{file_path}' with 'TyreClass' column.")
                                            else:
                                                print(f"    Skipping '{file_path}': No compound allocation found for {current_track_enum.value} in {current_year}.")
                                        else:
                                            print(f"    Skipping '{file_path}': Empty or missing 'Compound' column.")
                                    else:
                                        print(f"    Skipping '{file_path}': Filename format not recognized (expected YEAR_SESSIONTYPE.xlsx).")

                                except Exception as e:
                                    print(f"    Error processing '{file_path}': {e}")
            else:
                print(f"  Skipping non-directory item in {input_track_dir}: {team_name}") 

print(f"\n--- Finished adding TyreClass column to all relevant Excel files across all specified tracks. ---")

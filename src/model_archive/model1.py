import os
import numpy as np
import pandas as pd
from constants import TyreCompound, Track 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

output_base_dir = "Data_Suzuka"
session_track = Track.SUZUKA


SUZUKA_RACE_LAPS = 53 #53 FOR SUZUKA, 57 FOR MIAMI
AVG_PIT_STOP_TIME_SECONDS = 22.5

print(f"--- Loading Data from '{output_base_dir}' for Linear Regression Model ---")

# reading from Excel files
all_laps_df = pd.DataFrame()
race_sessions_found = 0
total_laps_loaded = 0

# Walk through the directory structure: Team -> Driver -> Session.xlsx
for team_name in os.listdir(output_base_dir):
    team_dir = os.path.join(output_base_dir, team_name)
    if os.path.isdir(team_dir):
        for driver_name in os.listdir(team_dir):
            driver_dir = os.path.join(team_dir, driver_name)
            if os.path.isdir(driver_dir):
                for file_name in os.listdir(driver_dir):
                    if file_name.endswith(".xlsx") and 'RACE' in file_name.upper(): # Only load RACE sessions
                        file_path = os.path.join(driver_dir, file_name)
                        try:
                            # Read the Excel file into a DataFrame
                            df = pd.read_excel(file_path)
                            if not df.empty:
                                all_laps_df = pd.concat([all_laps_df, df], ignore_index=True)
                                race_sessions_found += 1
                                total_laps_loaded += len(df)
                        except Exception as e:
                            print(f"  Error loading '{file_path}': {e}")

if all_laps_df.empty:
    print("No race data loaded. Cannot proceed with prediction.")
else:
    print(f"\nSuccessfully consolidated data for modeling.")

    # Preprocessing
    model_data = all_laps_df.dropna(subset=['LapTime', 'TrackTemp', 'Rain', 'Compound', 'Lap']).copy()

    if model_data.empty:
        print("After dropping NaNs, no valid race data remains.")
    else:
        print(f"After dropping NaNs, {len(model_data)} valid race laps remain for training.")
        # Features (X) and Target (y)
        features = ['Rain', 'TrackTemp', 'Compound', 'Lap']
        target = 'LapTime'

        X = model_data[features]
        y = model_data[target]

        categorical_features = ['Compound']
        numerical_features = ['Rain', 'TrackTemp', 'Lap']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough' # Keep numerical features as they are
        )

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training data size: {len(X_train)} laps")
        print(f"Testing data size: {len(X_test)} laps")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Evaluation:")
        print(f"  Mean Absolute Error (MAE): {mae:.3f} seconds")
        print(f"  R-squared (R2): {r2:.3f}")

        # --- New: Simulate Optimal Tyre Strategy for 2025 Race ---
        print("\n\n--- Simulating Optimal Tyre Strategy for 2025 Suzuka Race ---")

        hypothetical_track_temp = 30.0 # degrees Celsius  #40 FOR MIAMI
        hypothetical_rain = 0           # 0 for no rain, 1 for rain

        # Helper function to calculate total time for a stint
        def calculate_stint_time(start_lap, end_lap, compound, track_temp, rain, model, features):
            stint_laps = list(range(start_lap, end_lap + 1))
            stint_data = []
            for lap in stint_laps:
                stint_data.append({
                    'Rain': rain,
                    'TrackTemp': track_temp,
                    'Compound': compound,
                    'Lap': lap
                })
            stint_df = pd.DataFrame(stint_data)
            predicted_lap_times = model.predict(stint_df[features])
            return sum(predicted_lap_times)

        # Define possible tyre compounds for race (excluding UNKNOWN)
        available_compounds = [c.value for c in TyreCompound if c.value != 'UNKNOWN']


        simulated_strategies = []

        # --- 1-Stop Strategies ---
        stint_length_1_stop = SUZUKA_RACE_LAPS // 2

        for compound1 in available_compounds:
            for compound2 in available_compounds:
                if compound1 == compound2:
                    continue # Skip strategies where both stints are on the exact same compound

                strategy_name = f"1-Stop: {compound1} ({stint_length_1_stop} laps) -> {compound2} ({SUZUKA_RACE_LAPS - stint_length_1_stop} laps)"
                simulated_strategies.append({
                    'name': strategy_name,
                    'stints': [
                        {'compound': compound1, 'start_lap': 1, 'end_lap': stint_length_1_stop},
                        {'compound': compound2, 'start_lap': stint_length_1_stop + 1, 'end_lap': SUZUKA_RACE_LAPS}
                    ],
                    'pit_stops': 1
                })

        # --- 2-Stop Strategies ---
        stint_length_2_stop = SUZUKA_RACE_LAPS // 3

        for compound1 in available_compounds:
            for compound2 in available_compounds:
                for compound3 in available_compounds:
                    # Avoid identical consecutive compounds for simplicity, though possible in reality
                    if compound1 == compound2 or compound2 == compound3:
                        continue

                    strategy_name = f"2-Stop: {compound1} ({stint_length_2_stop} laps) -> {compound2} ({stint_length_2_stop} laps) -> {compound3} ({SUZUKA_RACE_LAPS - 2*stint_length_2_stop} laps)"
                    simulated_strategies.append({
                        'name': strategy_name,
                        'stints': [
                            {'compound': compound1, 'start_lap': 1, 'end_lap': stint_length_2_stop},
                            {'compound': compound2, 'start_lap': stint_length_2_stop + 1, 'end_lap': 2*stint_length_2_stop},
                            {'compound': compound3, 'start_lap': 2*stint_length_2_stop + 1, 'end_lap': SUZUKA_RACE_LAPS}
                        ],
                        'pit_stops': 2
                    })

        # Calculate total race time for each strategy
        strategy_results = []

        for strategy in simulated_strategies:
            total_race_time = 0
            for stint in strategy['stints']:
                stint_time = calculate_stint_time(
                    stint['start_lap'],
                    stint['end_lap'],
                    stint['compound'],
                    hypothetical_track_temp,
                    hypothetical_rain,
                    model,
                    features
                )
                total_race_time += stint_time
            total_race_time += strategy['pit_stops'] * AVG_PIT_STOP_TIME_SECONDS
            strategy_results.append({'Strategy': strategy['name'], 'TotalRaceTime': total_race_time})

        results_df = pd.DataFrame(strategy_results)
        results_df = results_df.sort_values(by='TotalRaceTime').reset_index(drop=True)

        print("\nHypothetical 2025 Suzuka Race Conditions:")
        print(f"  Race Laps: {SUZUKA_RACE_LAPS}")
        print(f"  Track Temperature: {hypothetical_track_temp}°C")
        print(f"  Rain: {'Yes' if hypothetical_rain == 1 else 'No'}")
        print(f"  Average Pit Stop Time: {AVG_PIT_STOP_TIME_SECONDS} seconds")

        print("\nSimulated Race Strategies and Predicted Total Race Times:")
        print(results_df.round(3))

        optimal_strategy = results_df.iloc[0]
        print(f"\nOptimal Tyre Strategy for 2025 Suzuka Race:")
        print(f"  Strategy: {optimal_strategy['Strategy']}")
        print(f"  Predicted Total Race Time: {optimal_strategy['TotalRaceTime']:.3f} seconds")

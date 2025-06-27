import os
import numpy as np
import pandas as pd
from enum import Enum # Added for demonstration, assuming constants.py exists
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from constants import TyreCompound
from track_data import TRACK_RACE_DATA

# --- Mapping for model's internal TyreClass strings ---
compound_to_tyre_class_map = {
    TyreCompound.SOFT.value: 'SLICK_SOFT',
    TyreCompound.MEDIUM.value: 'SLICK_MEDIUM',
    TyreCompound.HARD.value: 'SLICK_HARD',
    TyreCompound.INTER.value: 'INTERMEDIATE', # Using INTER as per provided code
    TyreCompound.WET.value: 'WET'
}
# For output readability, we also need a reverse map
tyre_class_to_compound_display_map = {v: k for k, v in compound_to_tyre_class_map.items()}

output_summary_file = "model_evaluation_summary_3_3_trainings.txt"
if os.path.exists(output_summary_file):
    os.remove(output_summary_file)
    print(f"Cleared existing summary file: {output_summary_file}")


# Helper function for Stint Time Calculation
def calculate_stint_time(start_lap, end_lap, tyre_class, track_temp, rain, current_model, features_list):
    stint_laps = list(range(start_lap, end_lap + 1))
    if not stint_laps:
        return 0

    stint_data = []
    for lap in stint_laps:
        stint_data.append({
            'Rain': rain,
            'TrackTemp': track_temp,
            'TyreClass': tyre_class,
            'Lap': lap
        })
    stint_df = pd.DataFrame(stint_data)
    predicted_lap_times = current_model.predict(stint_df[features_list])
    return sum(predicted_lap_times)

# --- NEW: Helper function to format lap ranges for display ---
def format_lap_interval_display(lap_number, total_race_laps):
    """
    Formats a single lap number into a 6-lap interval (lap +/- 3),
    clamped by 1 and total_race_laps.
    Returns a string like "LXX-YY".
    """
    start_lap = max(1, lap_number - 3)
    end_lap = min(total_race_laps, lap_number + 3)
    return f"L{start_lap}-{end_lap}"


# --- Main Loop to Process Each Track ---
for track_enum, track_details in TRACK_RACE_DATA.items():
    current_track_name = track_enum.name.title() # e.g., "Bahrain" from "BAHRAIN"
    output_base_dir = os.path.join("data", f"Data_{current_track_name}")

    print(f"\n{'='*80}")
    print(f"--- Processing Track: {current_track_name} ---")
    print(f"{'='*80}\n")

    # Get track-specific parameters
    RACE_LAPS = track_details["race_laps"]
    AVG_PIT_STOP_TIME_SECONDS = track_details["avg_pit_stop_time_seconds"]
    hypothetical_track_temp = track_details["track_temp_2025_celsius"]
    hypothetical_rain = 0 # Assuming 2025 simulations are for dry conditions unless specific rain allocation is added

    # Determine available 2025 tyres for simulation based on track_data
    compounds_2025 = track_details.get(2025, {}).get("compound_allocation", {})
    if not compounds_2025:
        print(f"No 2025 compound allocation found for {current_track_name}. Skipping simulation for this track.")
        continue # Skip to next track if no 2025 data

    available_tyre_classes_for_simulation = []
    for tyre_compound_enum, _ in compounds_2025.items():
        # Ensure we map from the Enum member's VALUE (e.g., "SOFT") to the string TyreClass (e.g., "SLICK_SOFT")
        model_tyre_class_string = compound_to_tyre_class_map.get(tyre_compound_enum.value)
        if model_tyre_class_string:
            available_tyre_classes_for_simulation.append(model_tyre_class_string)
    
    if not available_tyre_classes_for_simulation:
        print(f"No valid tyre classes derived for 2025 simulation for {current_track_name}. Skipping.")
        continue

    print(f"Race Laps: {RACE_LAPS}")
    print(f"Average Pit Stop Time: {AVG_PIT_STOP_TIME_SECONDS} seconds")
    print(f"Hypothetical 2025 Track Temperature: {hypothetical_track_temp}°C")
    print(f"Simulating a DRY race. Available Tyre Classes for simulation: {', '.join(available_tyre_classes_for_simulation)}")


    # --- Data Loading ---
    all_laps_df = pd.DataFrame()
    race_sessions_found = 0
    total_laps_loaded = 0

    print(f"\n--- Loading Race Data from '{output_base_dir}' ---")

    if not os.path.exists(output_base_dir):
        print(f"Data directory '{output_base_dir}' not found. Skipping {current_track_name}.")
        continue

    for team_name in os.listdir(output_base_dir):
        team_dir = os.path.join(output_base_dir, team_name)
        if os.path.isdir(team_dir):
            for driver_name in os.listdir(team_dir):
                driver_dir = os.path.join(team_dir, driver_name)
                if os.path.isdir(driver_dir):
                    for file_name in os.listdir(driver_dir):
                        if file_name.endswith(".xlsx"):# and 'RACE' in file_name.upper():
                            file_path = os.path.join(driver_dir, file_name)
                            try:
                                df = pd.read_excel(file_path)
                                if not df.empty:
                                    all_laps_df = pd.concat([all_laps_df, df], ignore_index=True)
                                    race_sessions_found += 1
                                    total_laps_loaded += len(df)
                            except Exception as e:
                                print(f"    Error loading '{file_path}': {e}")

    if all_laps_df.empty:
        print(f"No race data loaded for {current_track_name}. Cannot proceed with prediction.")
        continue # Skip to next track

    print(f"Successfully consolidated data for {current_track_name}. Loaded {total_laps_loaded} laps from {race_sessions_found} race sessions.")

    # --- Data Preprocessing ---
    required_columns = ['LapTime', 'TrackTemp', 'Rain', 'Compound', 'Lap']
    if not all(col in all_laps_df.columns for col in required_columns):
        print(f"Error: One or more required columns missing from data for {current_track_name}: {required_columns}")
        print("Available columns:", all_laps_df.columns.tolist())
        continue # Skip to next track
    
    model_data = all_laps_df.dropna(subset=required_columns).copy()

    if model_data.empty:
        print(f"After dropping NaNs, no valid race data remains for training for {current_track_name}.")
        continue # Skip to next track
    
    print(f"After dropping NaNs, {len(model_data)} valid race laps remain for training for {current_track_name}.")
    
    # --- Create TyreClass column (model's internal representation) ---
    model_data['TyreClass'] = model_data['Compound'].map(compound_to_tyre_class_map)
    model_data.dropna(subset=['TyreClass'], inplace=True) # Drop rows if compound map fails
    print(f"After mapping to TyreClass and dropping NaNs, {len(model_data)} valid laps remain for {current_track_name}.")

    if model_data.empty:
        print(f"After TyreClass mapping, no valid data remains for {current_track_name}. Skipping.")
        continue # Skip to next track

    # Features (X) and Target (y)
    features = ['Rain', 'TrackTemp', 'TyreClass', 'Lap']
    target = 'LapTime'

    X = model_data[features]
    y = model_data[target]

    categorical_features = ['TyreClass']
    numerical_features = ['Rain', 'TrackTemp', 'Lap']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- Define the models to evaluate ---
    models_to_evaluate = {
        "Linear Regression": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        "Random Forest Regressor": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        "K-Nearest Neighbors Regressor": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=5, n_jobs=-1))
        ]),
        "Support Vector Regressor": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR(kernel='rbf', C=100, epsilon=0.1))
        ]),
        "Decision Tree Regressor": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42))
        ]),
        "Gradient Boosting Regressor": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining data size for {current_track_name}: {len(X_train)} laps")
    print(f"Testing data size for {current_track_name}: {len(X_test)} laps")

    best_model_test_mae = float('inf')
    overall_best_simulation_time = float('inf')
    overall_best_strategy_info = {}
    
    full_model_evaluation_summary = []

    # --- Evaluate Each Model and Simulate Strategies ---
    for model_name, model_pipeline in models_to_evaluate.items():
        print(f"\n--- Training and Evaluating: {model_name} for {current_track_name} ---")
        model_pipeline.fit(X_train, y_train)
        
        # Predictions for evaluation
        y_train_pred = model_pipeline.predict(X_train)
        y_test_pred = model_pipeline.predict(X_test)
        
        # Calculate metrics for training set
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        # Calculate metrics for testing set
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        print(f"    Model Performance for {model_name}:")
        print(f"      Training R-squared (Correctness): {r2_train:.3f}")
        print(f"      Training Mean Absolute Error (MAE): {mae_train:.3f} seconds")
        print(f"      Testing R-squared (Correctness): {r2_test:.3f}")
        print(f"      Testing Mean Absolute Error (MAE): {mae_test:.3f} seconds")
        print(f"    Explanation of R-squared (Correctness):")
        print(f"      An R-squared value of {r2_test:.3f} on the testing set means the model explains {r2_test:.1%} of the variation in lap times.")
        print(f"      A value closer to 1.0 indicates a better fit and higher predictive power.")

        if mae_test < best_model_test_mae:
            best_model_test_mae = mae_test
            best_model_name_for_MAE = model_name

        # --- Generate and Evaluate Race Strategies for the current model ---
        simulated_strategies = []

        # 1-Stop Strategies
        min_stint_len_1 = 10 
        for stint1_end_lap in range(min_stint_len_1, RACE_LAPS - min_stint_len_1 + 1, 4):
            stint2_start_lap = stint1_end_lap + 1
            if (RACE_LAPS - stint1_end_lap) < min_stint_len_1:
                continue

            for tyre_class1 in available_tyre_classes_for_simulation:
                for tyre_class2 in available_tyre_classes_for_simulation:
                    if tyre_class1 == tyre_class2:
                        continue 
                    
                    compound1_display = tyre_class_to_compound_display_map.get(tyre_class1, tyre_class1)
                    compound2_display = tyre_class_to_compound_display_map.get(tyre_class2, tyre_class2)
                    
                    # Apply lap interval formatting to the pit stop lap (stint1_end_lap)
                    # The start of the second stint (stint2_start_lap) is exact, as it's the lap after the pitstop
                    formatted_stint1_end_lap_display = format_lap_interval_display(stint1_end_lap, RACE_LAPS)
                    
                    strategy_name = (
                        f"1-Stop: {compound1_display} (L1-approx {formatted_stint1_end_lap_display}) -> "
                        f"{compound2_display} (L{stint2_start_lap}-{RACE_LAPS})"
                    )
                    simulated_strategies.append({
                        'name': strategy_name,
                        'stints': [
                            {'tyre_class': tyre_class1, 'start_lap': 1, 'end_lap': stint1_end_lap},
                            {'tyre_class': tyre_class2, 'start_lap': stint2_start_lap, 'end_lap': RACE_LAPS}
                        ],
                        'pit_stops': 1
                    })

        # 2-Stop Strategies
        min_stint_len_2 = 8 
        for stint1_end_lap in range(min_stint_len_2, RACE_LAPS - 2 * min_stint_len_2 + 1, 4):
            for stint2_end_lap in range(stint1_end_lap + min_stint_len_2, RACE_LAPS - min_stint_len_2 + 1, 4):
                
                stint1_len = stint1_end_lap
                stint2_start_lap = stint1_end_lap + 1
                stint2_len = stint2_end_lap - stint1_end_lap
                stint3_start_lap = stint2_end_lap + 1
                stint3_len = RACE_LAPS - stint2_end_lap

                if stint1_len < min_stint_len_2 or stint2_len < min_stint_len_2 or stint3_len < min_stint_len_2:
                    continue

                for tyre_class1 in available_tyre_classes_for_simulation:
                    for tyre_class2 in available_tyre_classes_for_simulation:
                        for tyre_class3 in available_tyre_classes_for_simulation:
                            if tyre_class1 == tyre_class2 or tyre_class2 == tyre_class3:
                                continue
                            
                            compound1_display = tyre_class_to_compound_display_map.get(tyre_class1, tyre_class1)
                            compound2_display = tyre_class_to_compound_display_map.get(tyre_class2, tyre_class2)
                            compound3_display = tyre_class_to_compound_display_map.get(tyre_class3, tyre_class3)

                            # Apply lap interval formatting to the pit stop laps (stint1_end_lap, stint2_end_lap)
                            formatted_stint1_end_lap_display = format_lap_interval_display(stint1_end_lap, RACE_LAPS)
                            formatted_stint2_end_lap_display = format_lap_interval_display(stint2_end_lap, RACE_LAPS)

                            strategy_name = (
                                f"2-Stop: {compound1_display} (L1-approx {formatted_stint1_end_lap_display}) -> "
                                f"{compound2_display} (L{stint2_start_lap}-approx {formatted_stint2_end_lap_display}) -> "
                                f"{compound3_display} (L{stint3_start_lap}-{RACE_LAPS})"
                            )
                            simulated_strategies.append({
                                'name': strategy_name,
                                'stints': [
                                    {'tyre_class': tyre_class1, 'start_lap': 1, 'end_lap': stint1_end_lap},
                                    {'tyre_class': tyre_class2, 'start_lap': stint2_start_lap, 'end_lap': stint2_end_lap},
                                    {'tyre_class': tyre_class3, 'start_lap': stint3_start_lap, 'end_lap': RACE_LAPS}
                                ],
                                'pit_stops': 2
                            })

        print(f"    Total strategies generated for {model_name}: {len(simulated_strategies)}")

        strategy_results_for_current_model = []
        for strategy in simulated_strategies:
            total_race_time = 0
            for stint in strategy['stints']:
                stint_time = calculate_stint_time(
                    stint['start_lap'],
                    stint['end_lap'],
                    stint['tyre_class'],
                    hypothetical_track_temp,
                    hypothetical_rain,
                    model_pipeline,
                    features
                )
                total_race_time += stint_time
            total_race_time += strategy['pit_stops'] * AVG_PIT_STOP_TIME_SECONDS
            strategy_results_for_current_model.append({'Strategy': strategy['name'], 'TotalRaceTime': total_race_time})

        current_model_results_df = pd.DataFrame(strategy_results_for_current_model)
        
        # --- Store and Print Top 3 Strategies ---
        if not current_model_results_df.empty:
            current_model_results_df = current_model_results_df.sort_values(by='TotalRaceTime').reset_index(drop=True)
            
            top_3_strategies = current_model_results_df.head(3)
            # Use the actual best strategy for the 'overall best' tracking
            optimal_strategy_for_model = current_model_results_df.iloc[0] 

            print(f"\n    Top 3 Optimal Strategies for {model_name} on {current_track_name}:")
            # Prepare a list of strategy strings for the summary table
            summary_strategies_list = []
            for i, row in top_3_strategies.iterrows():
                strategy_line = f"{i+1}. Strategy: {row['Strategy']}, Time: {row['TotalRaceTime']:.3f}s"
                print(f"      {strategy_line}")
                summary_strategies_list.append(strategy_line)
            
            # Join the top 3 strategies into a single string for the summary table column
            top_strategies_combined_str = "\n".join(summary_strategies_list)

            full_model_evaluation_summary.append({
                'Model': model_name,
                'Training R2': r2_train,
                'Testing R2': r2_test,
                'Training MAE': mae_train,
                'Testing MAE': mae_test,
                'Optimal Strategy': top_strategies_combined_str, # Store the top 3 here
                'Predicted Race Time (s)': optimal_strategy_for_model['TotalRaceTime'] # Still the single best time
            })

            if optimal_strategy_for_model['TotalRaceTime'] < overall_best_simulation_time:
                overall_best_simulation_time = optimal_strategy_for_model['TotalRaceTime']
                overall_best_strategy_info = {
                    'Track': current_track_name,
                    'Model': model_name,
                    'Optimal Strategy': optimal_strategy_for_model['Strategy'], # Keep single best for overall
                    'Predicted Race Time': optimal_strategy_for_model['TotalRaceTime'],
                    'Model Test MAE': mae_test
                }
        else:
            print(f"    No valid strategies found for {model_name} on {current_track_name} given the simulation constraints.")
            full_model_evaluation_summary.append({
                'Model': model_name,
                'Training R2': r2_train,
                'Testing R2': r2_test,
                'Training MAE': mae_train,
                'Testing MAE': mae_test,
                'Optimal Strategy': "N/A (No strategies found)",
                'Predicted Race Time (s)': np.nan
            })

    print(f"\n\n--- Overall Best Strategy for {current_track_name} Across All Models ---")
    if overall_best_strategy_info:
        print(f"Hypothetical 2025 {current_track_name} Race Conditions:")
        print(f"    Race Laps: {RACE_LAPS}")
        print(f"    Track Temperature: {hypothetical_track_temp}°C")
        print(f"    Rain: {'Yes' if hypothetical_rain == 1 else 'No'}")
        print(f"    Average Pit Stop Time: {AVG_PIT_STOP_TIME_SECONDS} seconds")

        print(f"\nOverall Optimal Tyre Strategy for 2025 {current_track_name} Race:")
        print(f"    Model Used: {overall_best_strategy_info['Model']}")
        print(f"    Its Test MAE: {overall_best_strategy_info['Model Test MAE']:.3f} seconds")
        print(f"    Strategy: {overall_best_strategy_info['Optimal Strategy']}")
        print(f"    Predicted Total Race Time: {overall_best_strategy_info['Predicted Race Time']:.3f} seconds")
    else:
        print(f"No optimal strategy could be determined across any models for {current_track_name} under the given simulation conditions.")

    # --- Final Summary Table for Current Track ---
    print(f"\n\n--- Full Model Evaluation Summary for {current_track_name} ---")
    summary_df = pd.DataFrame(full_model_evaluation_summary)
    
    summary_df['Training R2'] = summary_df['Training R2'].map("{:.3f}".format)
    summary_df['Testing R2'] = summary_df['Testing R2'].map("{:.3f}".format)
    summary_df['Training MAE'] = summary_df['Training MAE'].map("{:.3f}".format)
    summary_df['Testing MAE'] = summary_df['Testing MAE'].map("{:.3f}".format)
    summary_df['Predicted Race Time (s)'] = summary_df['Predicted Race Time (s)'].apply(lambda x: "{:.3f}".format(x) if pd.notna(x) else "N/A")

    print(summary_df.to_string(index=False))
    summary_string = summary_df.to_string(index=False)

    # Append to the output file
    with open(output_summary_file, 'a') as f:
        f.write(f"\n\n--- Full Model Evaluation Summary for Track: {current_track_name} ---\n")
        f.write(summary_string)
        f.write(f"\n{'='*80}\n\n")

    print(f"\n{'='*80}")
    print(f"--- Finished Processing Track: {current_track_name} ---")
    print(f"{'='*80}\n\n")

print("\n--- All Tracks Processed ---")
print(f"\nAll model evaluation summaries have been saved to '{output_summary_file}'")
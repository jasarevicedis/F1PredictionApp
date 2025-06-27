import os
import numpy as np
import pandas as pd
from enum import Enum 
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

from track_data import TRACK_RACE_DATA


output_summary_file = "model_evaluation_summary_3_5.txt"
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

def format_lap_interval_display(lap_number, total_race_laps, interval_size=6):
    """
    Formates a single lap number into an interval (lap +/- half_interval_size),
    clamped by 1 and total_race_laps.
    Returns a string like "LXX-YY".
    """
    half_interval = interval_size // 2
    start_lap = max(1, lap_number - half_interval)
    end_lap = min(total_race_laps, lap_number + half_interval)
    return f"L{start_lap}-{end_lap}"


# --- Global setting for controlling the number of strategies ---
# Increasing this value will reduce the total number of strategies generated.
PIT_STOP_LAP_STEP_SIZE = 5 


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
    # This now correctly extracts the TyreClass ENUMS for simulation
    compounds_2025_allocation = track_details.get(2025, {}).get("compound_allocation", {})
    
    # Build a display map for the current track's 2025 tyre allocation
    # e.g., {'C1': 'HARD', 'C2': 'MEDIUM', 'C3': 'SOFT'}
    current_track_display_map = {}
    available_tyre_classes_for_simulation = []
    for tyre_compound_enum, tyre_class_enum in compounds_2025_allocation.items():
        if tyre_class_enum and tyre_compound_enum: # Ensure both are valid enums
            current_track_display_map[tyre_class_enum.value] = tyre_compound_enum.value
            available_tyre_classes_for_simulation.append(tyre_class_enum.value)
    
    if not available_tyre_classes_for_simulation:
        print(f"No valid 2025 compound allocation found or derivable for {current_track_name}. Skipping simulation for this track.")
        continue # Skip to next track if no 2025 data

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
                        if file_name.endswith(".xlsx") and 'RACE' in file_name.upper():
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
    # Now expecting 'TyreClass' directly in the input data, not 'Compound'
    required_columns = ['LapTime', 'TrackTemp', 'Rain', 'TyreClass', 'Lap'] 
    if not all(col in all_laps_df.columns for col in required_columns):
        print(f"Error: One or more required columns missing from data for {current_track_name}: {required_columns}")
        print("Available columns:", all_laps_df.columns.tolist())
        continue # Skip to next track
    
    model_data = all_laps_df.dropna(subset=required_columns).copy()

    if model_data.empty:
        print(f"After dropping NaNs, no valid race data remains for training for {current_track_name}.")
        continue # Skip to next track
    
    print(f"After dropping NaNs, {len(model_data)} valid race laps remain for training for {current_track_name}.")
    
    # The 'TyreClass' column is now assumed to be present and correctly formatted in the input data.
    # No mapping from 'Compound' to 'TyreClass' is needed here.
    # We only drop rows if TyreClass itself is missing.
    model_data.dropna(subset=['TyreClass'], inplace=True) 
    print(f"After ensuring valid TyreClass data, {len(model_data)} valid laps remain for {current_track_name}.")

    if model_data.empty:
        print(f"After TyreClass check, no valid data remains for {current_track_name}. Skipping.")
        continue # Skip to next track

    # Features (X) and Target (y)
    features = ['Rain', 'TrackTemp', 'TyreClass', 'Lap']
    target = 'LapTime'

    X = model_data[features]
    y = model_data[target]

    categorical_features = ['TyreClass'] # This column is now directly used as categorical feature
    numerical_features = ['Rain', 'TrackTemp', 'Lap']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    models_to_evaluate = {
        "Random Forest Regressor_100": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        "Random Forest Regressor_200": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        ]),
        "Random Forest Regressor_500": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1))
        ]),
        "K-Nearest Neighbors Regressor_5": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=5, n_jobs=-1))
        ]),
        "K-Nearest Neighbors Regressor_10": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=10, n_jobs=-1))
        ]),
        "K-Nearest Neighbors Regressor_15": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=15, n_jobs=-1))
        ]),
        "Decision Tree Regressor": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42))
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

        # Define an explicit hardness ranking for slick compounds (lower rank = harder compound)
        # Wet and Intermediate are not included here as they have different rules / are for specific conditions
        compound_ranking = {'HARD': 0, 'MEDIUM': 1, 'SOFT': 2}

        # 1-Stop Strategies
        min_stint_len_1 = 10 
        for stint1_end_lap in range(min_stint_len_1, RACE_LAPS - min_stint_len_1 + 1, PIT_STOP_LAP_STEP_SIZE):
            stint2_start_lap = stint1_end_lap + 1
            if (RACE_LAPS - stint1_end_lap) < min_stint_len_1:
                continue

            for tyre_class1_val in available_tyre_classes_for_simulation:
                for tyre_class2_val in available_tyre_classes_for_simulation:
                    # Rule: Consecutive stints must use different tire classes
                    if tyre_class1_val == tyre_class2_val:
                        continue 
                    
                    stint1_len = stint1_end_lap - 1 + 1 # (end_lap - start_lap + 1)
                    stint2_len = RACE_LAPS - stint2_start_lap + 1

                    # Get the compound type for each tire class used in this strategy
                    compound1_type = current_track_display_map.get(tyre_class1_val)
                    compound2_type = current_track_display_map.get(tyre_class2_val)

                    is_valid_stint_length_order = True
                    # Only apply rules to slick tires. Wet/Intermediate are excluded from hardness comparison.
                    if compound1_type in compound_ranking and compound2_type in compound_ranking:
                        rank1 = compound_ranking[compound1_type]
                        rank2 = compound_ranking[compound2_type]

                        # If tire1 is harder than tire2 (rank1 < rank2), then stint1_len should be >= stint2_len
                        if rank1 < rank2 and stint1_len < stint2_len:
                            is_valid_stint_length_order = False
                        # If tire2 is harder than tire1 (rank2 < rank1), then stint2_len should be >= stint1_len
                        elif rank2 < rank1 and stint2_len < stint1_len:
                            is_valid_stint_length_order = False
                    
                    if not is_valid_stint_length_order:
                        continue # Skip this strategy as it violates the hardness-stint length rule
                    
                    # Use the current_track_display_map for output formatting
                    compound1_display = current_track_display_map.get(tyre_class1_val, tyre_class1_val)
                    compound2_display = current_track_display_map.get(tyre_class2_val, tyre_class2_val)
                    
                    # Apply lap interval formatting to the pit stop lap (stint1_end_lap)
                    formatted_stint1_end_lap_display = format_lap_interval_display(stint1_end_lap, RACE_LAPS)
                    
                    strategy_name = (
                        f"1-Stop: {compound1_display} (L1-approx {formatted_stint1_end_lap_display}) -> "
                        f"{compound2_display} (L{stint2_start_lap}-{RACE_LAPS})"
                    )
                    simulated_strategies.append({
                        'name': strategy_name,
                        'stints': [
                            {'tyre_class': tyre_class1_val, 'start_lap': 1, 'end_lap': stint1_end_lap},
                            {'tyre_class': tyre_class2_val, 'start_lap': stint2_start_lap, 'end_lap': RACE_LAPS}
                        ],
                        'pit_stops': 1
                    })

        # 2-Stop Strategies
        min_stint_len_2 = 8 
        for stint1_end_lap in range(min_stint_len_2, RACE_LAPS - 2 * min_stint_len_2 + 1, PIT_STOP_LAP_STEP_SIZE):
            for stint2_end_lap in range(stint1_end_lap + min_stint_len_2, RACE_LAPS - min_stint_len_2 + 1, PIT_STOP_LAP_STEP_SIZE):
                
                stint1_len = stint1_end_lap - 1 + 1 # (end_lap - start_lap + 1)
                stint2_start_lap = stint1_end_lap + 1
                stint2_len = stint2_end_lap - stint1_end_lap
                stint3_start_lap = stint2_end_lap + 1
                stint3_len = RACE_LAPS - stint2_end_lap

                if stint1_len < min_stint_len_2 or stint2_len < min_stint_len_2 or stint3_len < min_stint_len_2:
                    continue

                for tyre_class1_val in available_tyre_classes_for_simulation:
                    for tyre_class2_val in available_tyre_classes_for_simulation:
                        for tyre_class3_val in available_tyre_classes_for_simulation:
                            # Rule: Consecutive stints must use different tire classes
                            if tyre_class1_val == tyre_class2_val or tyre_class2_val == tyre_class3_val:
                                continue
                            
                            # --- New logic to enforce realistic stint lengths based on tire hardness ---
                            stint_info = []
                            # Get the compound type for each tire class used in this strategy
                            compound1_type = current_track_display_map.get(tyre_class1_val)
                            compound2_type = current_track_display_map.get(tyre_class2_val)
                            compound3_type = current_track_display_map.get(tyre_class3_val)

                            # Add only slick tire stints to the list for comparison
                            if compound1_type in compound_ranking:
                                stint_info.append((compound1_type, stint1_len))
                            if compound2_type in compound_ranking:
                                stint_info.append((compound2_type, stint2_len))
                            if compound3_type in compound_ranking:
                                stint_info.append((compound3_type, stint3_len))
                            
                            is_valid_stint_length_order = True
                            # Compare all unique pairs of compounds within the current strategy's stints
                            for i in range(len(stint_info)):
                                for j in range(i + 1, len(stint_info)):
                                    type_a, len_a = stint_info[i]
                                    type_b, len_b = stint_info[j]

                                    rank_a = compound_ranking.get(type_a)
                                    rank_b = compound_ranking.get(type_b)

                                    # If compound A is harder than compound B (rank_a < rank_b), 
                                    # then A's stint length must be greater than or equal to B's.
                                    # If it's shorter (len_a < len_b), then this strategy is invalid.
                                    if rank_a < rank_b and len_a < len_b:
                                        is_valid_stint_length_order = False
                                        break
                                    # If compound B is harder than compound A (rank_b < rank_a),
                                    # then B's stint length must be greater than or equal to A's.
                                    # If it's shorter (len_b < len_a), then this strategy is invalid.
                                    elif rank_b < rank_a and len_b < len_a:
                                        is_valid_stint_length_order = False
                                        break
                                if not is_valid_stint_length_order:
                                    break
                            
                            if not is_valid_stint_length_order:
                                continue # Skip this strategy as it violates the hardness-stint length rule
                            # --- End of new logic ---
                            
                            # Use the current_track_display_map for output formatting
                            compound1_display = current_track_display_map.get(tyre_class1_val, tyre_class1_val)
                            compound2_display = current_track_display_map.get(tyre_class2_val, tyre_class2_val)
                            compound3_display = current_track_display_map.get(tyre_class3_val, tyre_class3_val)

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
                                    {'tyre_class': tyre_class1_val, 'start_lap': 1, 'end_lap': stint1_end_lap},
                                    {'tyre_class': tyre_class2_val, 'start_lap': stint2_start_lap, 'end_lap': stint2_end_lap},
                                    {'tyre_class': tyre_class3_val, 'start_lap': stint3_start_lap, 'end_lap': RACE_LAPS}
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
            
            # Get the top 2 overall strategies
            top_overall_2_strategies = current_model_results_df.head(2)

            # Find the best 2-stop strategy
            two_stop_strategies_df = current_model_results_df[current_model_results_df['Strategy'].str.contains('2-Stop:')].copy()
            
            best_2_stop_strategy = None
            if not two_stop_strategies_df.empty:
                best_2_stop_strategy = two_stop_strategies_df.iloc[0]

            # Combine strategies to form the final top 3 to display
            # Start with the top 2 overall
            final_top_strategies_list = top_overall_2_strategies.to_dict(orient='records')
            
            # Check if the overall top 3 needs adjustment to include a 2-stop
            if best_2_stop_strategy is not None:
                # Get the names of the currently selected top strategies
                current_top_names = [s['Strategy'] for s in final_top_strategies_list]

                # If the best 2-stop strategy is not already in the top 2 overall,
                # and we need a 3rd strategy, add it.
                if best_2_stop_strategy['Strategy'] not in current_top_names:
                    # If we only have 2 strategies, just add the best 2-stop as the 3rd.
                    if len(final_top_strategies_list) < 3:
                        final_top_strategies_list.append(best_2_stop_strategy.to_dict())
                    else:
                        # If we have 3, replace the 3rd overall with the best 2-stop.
                        # This might mean the 3rd overall was a 1-stop that is now replaced.
                        # We re-sort to ensure correct order after potential replacement.
                        final_top_strategies_list.append(best_2_stop_strategy.to_dict())
                        final_top_strategies_list = sorted(final_top_strategies_list, key=lambda x: x['TotalRaceTime'])
                        final_top_strategies_list = final_top_strategies_list[:3] # Keep only top 3 after sorting

            # Ensure we don't have duplicates and take exactly top 3 (or fewer if not enough generated)
            # Convert list of dicts to DataFrame to easily drop duplicates
            final_top_strategies_df = pd.DataFrame(final_top_strategies_list).drop_duplicates(subset=['Strategy']).sort_values(by='TotalRaceTime').head(3)
            
            # The actual best strategy for overall tracking (always the absolute best)
            optimal_strategy_for_model = current_model_results_df.iloc[0] 

            print(f"\n    Top 3 Optimal Strategies for {model_name} on {current_track_name}:")
            summary_strategies_list = []
            for i, row in final_top_strategies_df.iterrows(): # Iterate over the finalized top_3_strategies_df
                strategy_line = f"{i+1}. Strategy: {row['Strategy']}, Time: {row['TotalRaceTime']:.3f}s"
                print(f"      {strategy_line}")
                summary_strategies_list.append(strategy_line)
            
            top_strategies_combined_str = "\n".join(summary_strategies_list)

            full_model_evaluation_summary.append({
                'Model': model_name,
                'Training R2': r2_train,
                'Testing R2': r2_test,
                'Training MAE': mae_train,
                'Testing MAE': mae_test,
                'Optimal Strategy': top_strategies_combined_str, # Store the selected top strategies
                'Predicted Race Time (s)': optimal_strategy_for_model['TotalRaceTime'] # Still the single absolute best time
            })

            if optimal_strategy_for_model['TotalRaceTime'] < overall_best_simulation_time:
                overall_best_simulation_time = optimal_strategy_for_model['TotalRaceTime']
                overall_best_strategy_info = {
                    'Track': current_track_name,
                    'Model': model_name,
                    'Optimal Strategy': optimal_strategy_for_model['Strategy'], # Keep single absolute best for overall
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

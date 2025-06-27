import os
import numpy as np
import pandas as pd
from constants import TyreCompound, Track # Assuming these are defined elsewhere
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

# --- Configuration ---
output_base_dir = "data/Data_Bahrain"
session_track = Track.BAHRAIN

SUZUKA_RACE_LAPS = 57
AVG_PIT_STOP_TIME_SECONDS = 25

print(f"--- Loading Data from '{output_base_dir}' for Multiple Regression Models ---")

# --- Data Loading ---
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
                    # Only load RACE sessions as per your filter
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
    print("No race data loaded. Cannot proceed with prediction.")
else:
    print(f"\nSuccessfully consolidated data for modeling. Loaded {total_laps_loaded} laps from {race_sessions_found} race sessions.")

    # --- Data Preprocessing ---
    required_columns = ['LapTime', 'TrackTemp', 'Rain', 'Compound', 'Lap']
    if not all(col in all_laps_df.columns for col in required_columns):
        print(f"Error: One or more required columns missing from data: {required_columns}")
        print("Available columns:", all_laps_df.columns.tolist())
    else:
        model_data = all_laps_df.dropna(subset=required_columns).copy()

        if model_data.empty:
            print("After dropping NaNs, no valid race data remains for training.")
        else:
            print(f"After dropping NaNs, {len(model_data)} valid race laps remain for training.")
            
            # --- Create TyreClass column ---
            compound_to_tyre_class_map = {
                TyreCompound.SOFT.value: 'SLICK_SOFT',
                TyreCompound.MEDIUM.value: 'SLICK_MEDIUM',
                TyreCompound.HARD.value: 'SLICK_HARD',
                TyreCompound.INTER.value: 'INTERMEDIATE',
                TyreCompound.WET.value: 'WET'
            }
            # For output readability, we also need a reverse map
            tyre_class_to_compound_display_map = {v: k for k, v in compound_to_tyre_class_map.items()}

            model_data['TyreClass'] = model_data['Compound'].map(compound_to_tyre_class_map)
            # Drop rows where TyreClass might be NaN if an unknown compound appeared
            model_data.dropna(subset=['TyreClass'], inplace=True)
            print(f"After mapping to TyreClass and dropping NaNs, {len(model_data)} valid laps remain.")

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

            # --- Define the models to evaluate, including new ones ---
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

            print(f"\nTraining data size: {len(X_train)} laps")
            print(f"Testing data size: {len(X_test)} laps")

            best_model_test_mae = float('inf')
            overall_best_simulation_time = float('inf')
            overall_best_strategy_info = {}
            
            full_model_evaluation_summary = []

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

            # Simulation Conditions
            hypothetical_track_temp = 34.50
            hypothetical_rain = 0

            if hypothetical_rain == 0:
                available_tyre_classes_for_simulation = ['SLICK_SOFT', 'SLICK_MEDIUM', 'SLICK_HARD']
                print("\nSimulating a DRY race: Only SLICK_SOFT, SLICK_MEDIUM, SLICK_HARD tyre classes considered.")
            else:
                available_tyre_classes_for_simulation = ['INTERMEDIATE', 'WET']
                print("\nSimulating a WET race: Only INTERMEDIATE, WET tyre classes considered.")

            # --- Evaluate Each Model and Simulate Strategies ---
            for model_name, model_pipeline in models_to_evaluate.items():
                print(f"\n--- Training and Evaluating: {model_name} ---")
                model_pipeline.fit(X_train, y_train)
                
                # Predictions for evaluation
                y_train_pred = model_pipeline.predict(X_train)
                y_test_pred = model_pipeline.predict(X_test)
                
                mae_train = mean_absolute_error(y_train, y_train_pred)
                r2_train = r2_score(y_train, y_train_pred)

                mae_test = mean_absolute_error(y_test, y_test_pred)
                r2_test = r2_score(y_test, y_test_pred)

                print(f"  Model Performance:")
                print(f"    Training R-squared (Correctness): {r2_train:.3f}")
                print(f"    Training Mean Absolute Error (MAE): {mae_train:.3f} seconds")
                print(f"    Testing R-squared (Correctness): {r2_test:.3f}")
                print(f"    Testing Mean Absolute Error (MAE): {mae_test:.3f} seconds")
                print(f"  Explanation of R-squared (Correctness):")
                print(f"    An R-squared value of {r2_test:.3f} on the testing set means the model explains {r2_test:.1%} of the variation in lap times.")
                print(f"    A value closer to 1.0 indicates a better fit and higher predictive power.")


                if mae_test < best_model_test_mae:
                    best_model_test_mae = mae_test
                    best_model_name_for_MAE = model_name

                # --- Generate and Evaluate Race Strategies for the current model ---
                simulated_strategies = []

                # 1-Stop Strategies
                min_stint_len_1 = 10 
                for stint1_end_lap in range(min_stint_len_1, SUZUKA_RACE_LAPS - min_stint_len_1 + 1, 2):
                    stint2_start_lap = stint1_end_lap + 1
                    if (SUZUKA_RACE_LAPS - stint1_end_lap) < min_stint_len_1:
                        continue

                    for tyre_class1 in available_tyre_classes_for_simulation:
                        for tyre_class2 in available_tyre_classes_for_simulation:
                            if tyre_class1 == tyre_class2:
                                continue 

                            compound1_display = tyre_class_to_compound_display_map[tyre_class1]
                            compound2_display = tyre_class_to_compound_display_map[tyre_class2]
                            
                            strategy_name = f"1-Stop: {compound1_display} (L1-{stint1_end_lap}) -> {compound2_display} (L{stint2_start_lap}-{SUZUKA_RACE_LAPS})"
                            simulated_strategies.append({
                                'name': strategy_name,
                                'stints': [
                                    {'tyre_class': tyre_class1, 'start_lap': 1, 'end_lap': stint1_end_lap},
                                    {'tyre_class': tyre_class2, 'start_lap': stint2_start_lap, 'end_lap': SUZUKA_RACE_LAPS}
                                ],
                                'pit_stops': 1
                            })

                min_stint_len_2 = 8 
                for stint1_end_lap in range(min_stint_len_2, SUZUKA_RACE_LAPS - 2 * min_stint_len_2 + 1, 2):
                    for stint2_end_lap in range(stint1_end_lap + min_stint_len_2, SUZUKA_RACE_LAPS - min_stint_len_2 + 1, 2):
                        
                        stint1_len = stint1_end_lap
                        stint2_start_lap = stint1_end_lap + 1
                        stint2_len = stint2_end_lap - stint1_end_lap
                        stint3_start_lap = stint2_end_lap + 1
                        stint3_len = SUZUKA_RACE_LAPS - stint2_end_lap

                        if stint1_len < min_stint_len_2 or stint2_len < min_stint_len_2 or stint3_len < min_stint_len_2:
                            continue

                        for tyre_class1 in available_tyre_classes_for_simulation:
                            for tyre_class2 in available_tyre_classes_for_simulation:
                                for tyre_class3 in available_tyre_classes_for_simulation:
                                    if tyre_class1 == tyre_class2 or tyre_class2 == tyre_class3:
                                        continue
                                    
                                    compound1_display = tyre_class_to_compound_display_map[tyre_class1]
                                    compound2_display = tyre_class_to_compound_display_map[tyre_class2]
                                    compound3_display = tyre_class_to_compound_display_map[tyre_class3]

                                    strategy_name = f"2-Stop: {compound1_display} (L1-{stint1_end_lap}) -> {compound2_display} (L{stint2_start_lap}-{stint2_end_lap}) -> {compound3_display} (L{stint3_start_lap}-{SUZUKA_RACE_LAPS})"
                                    simulated_strategies.append({
                                        'name': strategy_name,
                                        'stints': [
                                            {'tyre_class': tyre_class1, 'start_lap': 1, 'end_lap': stint1_end_lap},
                                            {'tyre_class': tyre_class2, 'start_lap': stint2_start_lap, 'end_lap': stint2_end_lap},
                                            {'tyre_class': tyre_class3, 'start_lap': stint3_start_lap, 'end_lap': SUZUKA_RACE_LAPS}
                                        ],
                                        'pit_stops': 2
                                    })

                print(f"  Total strategies generated for {model_name}: {len(simulated_strategies)}")

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
                
                if not current_model_results_df.empty:
                    current_model_results_df = current_model_results_df.sort_values(by='TotalRaceTime').reset_index(drop=True)
                    optimal_strategy_for_model = current_model_results_df.iloc[0]
                    
                    print(f"\n  Optimal Strategy for {model_name}:")
                    print(f"    Strategy: {optimal_strategy_for_model['Strategy']}")
                    print(f"    Predicted Total Race Time: {optimal_strategy_for_model['TotalRaceTime']:.3f} seconds")
                    
                    full_model_evaluation_summary.append({
                        'Model': model_name,
                        'Training R2': r2_train,
                        'Testing R2': r2_test,
                        'Training MAE': mae_train,
                        'Testing MAE': mae_test,
                        'Optimal Strategy': optimal_strategy_for_model['Strategy'],
                        'Predicted Race Time (s)': optimal_strategy_for_model['TotalRaceTime']
                    })

                    if optimal_strategy_for_model['TotalRaceTime'] < overall_best_simulation_time:
                        overall_best_simulation_time = optimal_strategy_for_model['TotalRaceTime']
                        overall_best_strategy_info = {
                            'Model': model_name,
                            'Optimal Strategy': optimal_strategy_for_model['Strategy'],
                            'Predicted Race Time': optimal_strategy_for_model['TotalRaceTime'],
                            'Model Test MAE': mae_test
                        }
                else:
                    print(f"  No valid strategies found for {model_name} given the simulation constraints.")
                    full_model_evaluation_summary.append({
                        'Model': model_name,
                        'Training R2': r2_train,
                        'Testing R2': r2_test,
                        'Training MAE': mae_train,
                        'Testing MAE': mae_test,
                        'Optimal Strategy': "N/A (No strategies found)",
                        'Predicted Race Time (s)': np.nan
                    })


            print("\n\n--- Overall Best Strategy Across All Models ---")
            if overall_best_strategy_info:
                print(f"Hypothetical 2025 Suzuka Race Conditions:")
                print(f"  Race Laps: {SUZUKA_RACE_LAPS}")
                print(f"  Track Temperature: {hypothetical_track_temp}°C")
                print(f"  Rain: {'Yes' if hypothetical_rain == 1 else 'No'}")
                print(f"  Average Pit Stop Time: {AVG_PIT_STOP_TIME_SECONDS} seconds")

                print("\nOverall Optimal Tyre Strategy for 2025 Suzuka Race:")
                print(f"  Model Used: {overall_best_strategy_info['Model']}")
                print(f"  Its Test MAE: {overall_best_strategy_info['Model Test MAE']:.3f} seconds")
                print(f"  Strategy: {overall_best_strategy_info['Optimal Strategy']}")
                print(f"  Predicted Total Race Time: {overall_best_strategy_info['Predicted Race Time']:.3f} seconds")
            else:
                print("No optimal strategy could be determined across any models under the given simulation conditions.")

            print("\n\n--- Full Model Evaluation Summary ---")
            summary_df = pd.DataFrame(full_model_evaluation_summary)
            
            # Format numerical columns for better readability
            summary_df['Training R2'] = summary_df['Training R2'].map("{:.3f}".format)
            summary_df['Testing R2'] = summary_df['Testing R2'].map("{:.3f}".format)
            summary_df['Training MAE'] = summary_df['Training MAE'].map("{:.3f}".format)
            summary_df['Testing MAE'] = summary_df['Testing MAE'].map("{:.3f}".format)
            summary_df['Predicted Race Time (s)'] = summary_df['Predicted Race Time (s)'].apply(lambda x: "{:.3f}".format(x) if pd.notna(x) else "N/A")

            print(summary_df.to_string(index=False))
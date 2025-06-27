from constants import Track, TyreCompound, TyreClass
import math

TRACK_RACE_DATA = {
    Track.AUSTRALIA: {
        "lap_length_km": 5.303,
        "race_laps": 58,  
        "corners": 14,
        "high_speed_corners": 4,
        "low_speed_corners": 6,
        "average_speed_kph": 235,
        "downforce_level": "Medium",
        "overtaking_difficulty": "Medium",
        "avg_pit_stop_time_seconds": 13.5, 
        "track_temp_2025_celsius": 18, 
        2022: {
            "race_laps": 58, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C5}
        },
        2023: {
            "race_laps": 58, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2024: {
            "race_laps": 58, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2025: {
            "race_laps": 58, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        }
    },
    Track.CHINA: {
        "lap_length_km": 5.451,
        "race_laps": 56, 
        "corners": 16,
        "high_speed_corners": 3,
        "low_speed_corners": 8,
        "average_speed_kph": 210,
        "downforce_level": "High",
        "overtaking_difficulty": "Medium",
        "avg_pit_stop_time_seconds": 22.5, 
        "track_temp_2025_celsius": 42, 
        2022: {
            "race_laps": 0, # Race did not happen
            "compound_allocation": {}
        },
        2023: {
            "race_laps": 0, # Race did not happen
            "compound_allocation": {}
        },
        2024: {
            "race_laps": 56, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2025: {
            "race_laps": 56, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        }
    },
    Track.SUZUKA: {
        "lap_length_km": 5.807,
        "race_laps": 53, # Max laps for the track
        "corners": 18,
        "high_speed_corners": 6,
        "low_speed_corners": 4,
        "average_speed_kph": 230,
        "downforce_level": "High",
        "overtaking_difficulty": "Medium",
        "avg_pit_stop_time_seconds": 23.5, # Average pit lane time loss for Suzuka
        "track_temp_2025_celsius": 22.5, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 53, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2023: {
            "race_laps": 53, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2024: {
            "race_laps": 53, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2025: {
            "race_laps": 53, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        }
    },
    Track.BAHRAIN: {
        "lap_length_km": 5.412,
        "race_laps": 57, # Max laps for the track
        "corners": 15,
        "high_speed_corners": 2,
        "low_speed_corners": 6,
        "average_speed_kph": 220,
        "downforce_level": "Medium",
        "overtaking_difficulty": "Medium-High",
        "avg_pit_stop_time_seconds": 24, # Average pit lane time loss for Bahrain
        "track_temp_2025_celsius": 34.5, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 57, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2023: {
            "race_laps": 57, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2024: {
            "race_laps": 57, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2025: {
            "race_laps": 57, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        }
    },
    Track.JEDDAH: {
        "lap_length_km": 6.174,
        "race_laps": 50, # Max laps for the track
        "corners": 27,
        "high_speed_corners": 16,
        "low_speed_corners": 4,
        "average_speed_kph": 250,
        "downforce_level": "Low-Medium",
        "overtaking_difficulty": "Medium",
        "avg_pit_stop_time_seconds": 20, # Average pit lane time loss for Jeddah
        "track_temp_2025_celsius": 40, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 50, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2023: {
            "race_laps": 50, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2024: {
            "race_laps": 50, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2025: {
            "race_laps": 50, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        }
    },
    Track.MIAMI: {
        "lap_length_km": 5.410,
        "race_laps": 57, # Max laps for the track
        "corners": 19,
        "high_speed_corners": 5,
        "low_speed_corners": 8,
        "average_speed_kph": 215,
        "downforce_level": "Medium",
        "overtaking_difficulty": "Medium",
        "avg_pit_stop_time_seconds": 21, # Average pit lane time loss for Miami
        "track_temp_2025_celsius": 40, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 57, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2023: {
            "race_laps": 57, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2024: {
            "race_laps": 57, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2025: {
            "race_laps": 57, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        }
    },
    Track.IMOLA: {
        "lap_length_km": 4.909,
        "race_laps": 63, # Max laps for the track
        "corners": 19,
        "high_speed_corners": 5,
        "low_speed_corners": 9,
        "average_speed_kph": 220,
        "downforce_level": "Medium-High",
        "overtaking_difficulty": "Low",
        "avg_pit_stop_time_seconds": 29, # Average pit lane time loss for Imola
        "track_temp_2025_celsius": 45, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 63, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2023: {
            "race_laps": 0, # Race did not happen
            "compound_allocation": {}
        },
        2024: {
            "race_laps": 63, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2025: {
            "race_laps": 63, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        }
    },
    Track.MONACO: {
        "lap_length_km": 3.337,
        "race_laps": 78, # Max laps for the track
        "corners": 19,
        "high_speed_corners": 0,
        "low_speed_corners": 19,
        "average_speed_kph": 160,
        "downforce_level": "Very High",
        "overtaking_difficulty": "Very Low",
        "avg_pit_stop_time_seconds": 24, # Average pit lane time loss for Monaco
        "track_temp_2025_celsius": 42, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 78, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2023: {
            "race_laps": 78, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2024: {
            "race_laps": 78, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2025: {
            "race_laps": 78, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        }
    },
    Track.BARCELONA: {
        "lap_length_km": 4.655,
        "race_laps": 66, # Max laps for the track
        "corners": 14,
        "high_speed_corners": 5,
        "low_speed_corners": 5,
        "average_speed_kph": 210,
        "downforce_level": "High",
        "overtaking_difficulty": "Medium",
        "avg_pit_stop_time_seconds": 21.5, # Average pit lane time loss for Barcelona
        "track_temp_2025_celsius": 48, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 66, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2023: {
            "race_laps": 66, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2024: {
            "race_laps": 66, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        },
        2025: {
            "race_laps": 66, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C1, TyreCompound.MEDIUM: TyreClass.C2, TyreCompound.SOFT: TyreClass.C3}
        }
    },
    Track.CANADA: {
        "lap_length_km": 4.361,
        "race_laps": 70, # Max laps for the track
        "corners": 14,
        "high_speed_corners": 4,
        "low_speed_corners": 4,
        "average_speed_kph": 215,
        "downforce_level": "Medium-Low",
        "overtaking_difficulty": "Medium-High",
        "avg_pit_stop_time_seconds": 23, # Average pit lane time loss for Canada
        "track_temp_2025_celsius": 47, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 70, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2023: {
            "race_laps": 70, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2024: {
            "race_laps": 70, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2025: {
            "race_laps": 70, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        }
    },
    Track.AUSTRIA: {
        "lap_length_km": 4.318,
        "race_laps": 71, # Max laps for the track
        "corners": 10,
        "high_speed_corners": 6,
        "low_speed_corners": 2,
        "average_speed_kph": 230,
        "downforce_level": "Medium",
        "overtaking_difficulty": "Medium",
        "avg_pit_stop_time_seconds": 19.0, # Estimated average pit lane time loss for Austria
        "track_temp_2025_celsius": 38, # Estimated track temp for 2025 race
        2022: {
            "race_laps": 71, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C2, TyreCompound.MEDIUM: TyreClass.C3, TyreCompound.SOFT: TyreClass.C4}
        },
        2023: {
            "race_laps": 71, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2024: {
            "race_laps": 71, # Race happened
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        },
        2025: {
            "race_laps": 71, # Race is planned
            "compound_allocation": {TyreCompound.HARD: TyreClass.C3, TyreCompound.MEDIUM: TyreClass.C4, TyreCompound.SOFT: TyreClass.C5}
        }
    }
}
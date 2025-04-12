from enum import Enum

class Driver(str, Enum):
    MAX = "max"
    LEC = "lec"
    HAM = "ham"
    ALO = "alo"
    PER = "per"
    NOR = "nor"

class Track(str, Enum):
    MONACO = "Monaco"
    SPA = "Spa"
    MONZA = "Monza"
    SUZUKA = "Suzuka"
    ABU_DHABI = "Abu Dhabi"

class TyreCompound(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTER = "INTERMEDIATE"
    WET = "WET"

class SessionType(str, Enum):
    FP1 = "Practice 1"
    FP2 = "FP2"
    FP3 = "FP3"
    QUALIFYING = "Q"
    RACE = "R"


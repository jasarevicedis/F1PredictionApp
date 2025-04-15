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

class TyreClass(str, Enum):
    C1 = 'c1', #hardest
    C2 = 'c2',
    C3 = 'c3',
    C4 = 'c4',
    C5 = 'c5',
    C6 = 'c6',

class SessionType(str, Enum):
    FP1 = "Practice 1"
    FP2 = "FP2"
    FP3 = "FP3"
    QUALIFYING = "Q"
    RACE = "R"


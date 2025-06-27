from enum import Enum

class Driver(str, Enum):
    MAX = "ver"           #redbull
    TSUNODA = "tsu"       #redbull
    LECLERC = "lec"       #ferrari
    HAMILTON = "ham"      #ferrari
    ALONSO = "alo"        #aston martin
    STROLL = "str"        #aston martin
    NORRIS = "nor"        #mclaren
    PIASTRI = "pia"       #mclaren
    RUSSEL = "rus"        #mercedes
    ANTONELI = "ant"      #mercedes
    ALBON = "alb"         #williams
    SAINZ = "sai"         #williams
    COLAPINTO = "col"     #alpine
    DOOHAN = "doo"        #kick (third)
    BORTOLETO = "bor"     #kick
    HULKENBERG = "hul"    #kick
    LAWSON = "law"        #racingbulls
    HADJAR = "had"        #racingbulls
    OCON = "oco"          #haas
    BEARMAN = "bea"       #haas

class Track(str, Enum):
    AUSTRALIA = "Australia"
    CHINA = "China"
    SUZUKA = "Suzuka"
    BAHRAIN = "Bahrain"
    JEDDAH = "Jeddah"
    MIAMI = "Miami"
    IMOLA ="Imola"
    MONACO = "Monaco"
    BARCELONA = "Barcelona"
    CANADA = "Canada"
    AUSTRIA = "Austria"
    SILVERSTONE = "Silverstone"
    SPA = "Spa"
    HUNGARY = "Hungary"
    NETHERLANDS = "Netherlands"
    MONZA = "Monza"
    BAKU = "Baku"
    SINGAPORE = "Singapore"
    COTA = "Cota"
    MEXICO = "Mexico"
    BRAZIL = "Brazil"
    LASVEGAS = "Las Vegas"
    QATAR = "Qatar"
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
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    QUALIFYING = "Q"
    RACE = "Race"


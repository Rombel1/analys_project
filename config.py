"""
Конфигурация системы
"""

from pathlib import Path

# Базовые пути - исправляем на вашу реальную структуру
BASE_DIR = Path(__file__).parent  # C:/Users/user/Desktop/Python_projects/analys_project

# Возможные места, где могут лежать данные
POSSIBLE_DATA_PATHS = [
    BASE_DIR / "data",                                    # analys_project/data
    Path("C:/Users/user/Desktop/Python_projects/analys_project/data"),
]

# Находим существующую папку с данными
DATA_DIR = None
for path in POSSIBLE_DATA_PATHS:
    if path.exists() and any(path.glob("*.csv")):
        DATA_DIR = path
        break

if DATA_DIR is None:
    # Если не нашли, создаем в текущей папке
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    print(f" Создана папка для данных: {DATA_DIR}")
    print(f"   Поместите туда файлы cities_historical_2015_2024.csv и cities_forecast_2024_2039.csv")

REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

# Создаем директории
for dir_path in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Файлы с данными
HISTORICAL_FILE = DATA_DIR / "cities_historical_2015_2024.csv"
FORECAST_FILE = DATA_DIR / "cities_forecast_2024_2039.csv"

# Параметры прогнозирования
FORECAST_HORIZONS = [5, 10, 15]
FORECAST_CONFIDENCE = 0.95

# Коэффициенты для демографических показателей
DEMOGRAPHIC_RATES = {
    'birth_rate_high': 12.0,
    'birth_rate_medium': 9.0,
    'birth_rate_low': 6.0,
    'death_rate_high': 15.0,
    'death_rate_medium': 12.0,
    'death_rate_low': 9.0,
}

# Пороги для классификации городов
POPULATION_THRESHOLDS = {
    'million': 1_000_000,
    'large': 500_000,
    'medium': 250_000,
    'small': 100_000,
    'town': 50_000
}

print(f"\n Конфигурация загружена:")
print(f"   DATA_DIR: {DATA_DIR}")
print(f"   HISTORICAL_FILE: {HISTORICAL_FILE}")
print(f"   FORECAST_FILE: {FORECAST_FILE}")
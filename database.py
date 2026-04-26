"""
Модуль базы данных для хранения прогнозов
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "forecasts.db"


def get_connection():
    """Получить соединение с БД с проверкой"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")  # Улучшенная производительность
    return conn


def init_db():
    """Инициализация базы данных"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Создаем таблицу если её нет
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                forecast_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Создаем индекс если его нет
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_city ON saved_forecasts(city)
        """)
        
        # Проверяем что таблица создалась
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='saved_forecasts'")
        if cursor.fetchone():
            logger.info("База данных инициализирована успешно")
        else:
            logger.error("Не удалось создать таблицу saved_forecasts")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Ошибка инициализации БД: {e}")
        raise


def ensure_db():
    """Проверить что БД существует и инициализирована"""
    if not DB_PATH.exists():
        logger.info(f"Создание новой БД: {DB_PATH}")
        init_db()
    else:
        # Проверяем что таблица существует
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='saved_forecasts'")
            if not cursor.fetchone():
                logger.warning("Таблица не найдена, пересоздаем...")
                conn.close()
                # Удаляем поврежденный файл и создаем заново
                DB_PATH.unlink(missing_ok=True)
                init_db()
            else:
                conn.close()
        except Exception as e:
            logger.error(f"Ошибка проверки БД: {e}")
            conn.close()
            # Пробуем пересоздать
            DB_PATH.unlink(missing_ok=True)
            init_db()


def save_forecast(city: str, horizon: int, forecast_data: dict) -> int:
    """Сохранение прогноза в БД"""
    ensure_db()  # Проверяем БД перед использованием
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Проверяем, есть ли уже прогноз для этого города с таким горизонтом
        cursor.execute(
            "SELECT id FROM saved_forecasts WHERE city = ? AND horizon = ?",
            (city, horizon)
        )
        existing = cursor.fetchone()
        
        forecast_json = json.dumps(forecast_data, ensure_ascii=False, default=str)
        now = datetime.now().isoformat()
        
        if existing:
            # Обновляем существующий
            cursor.execute(
                "UPDATE saved_forecasts SET forecast_data = ?, updated_at = ? WHERE id = ?",
                (forecast_json, now, existing[0])
            )
            record_id = existing[0]
            logger.info(f"Прогноз для {city} обновлен (id={record_id})")
        else:
            # Создаем новый
            cursor.execute(
                "INSERT INTO saved_forecasts (city, horizon, forecast_data, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (city, horizon, forecast_json, now, now)
            )
            record_id = cursor.lastrowid
            logger.info(f"Прогноз для {city} сохранен (id={record_id})")
        
        conn.commit()
        conn.close()
        return record_id
        
    except sqlite3.OperationalError as e:
        logger.error(f"Ошибка SQLite при сохранении: {e}")
        # Пробуем пересоздать БД
        DB_PATH.unlink(missing_ok=True)
        init_db()
        # Пробуем снова
        return save_forecast(city, horizon, forecast_data)
    except Exception as e:
        logger.error(f"Ошибка сохранения прогноза: {e}")
        raise


def get_all_forecasts() -> list:
    """Получение всех сохраненных прогнозов"""
    ensure_db()
    
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM saved_forecasts ORDER BY updated_at DESC LIMIT 50"
        )
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except sqlite3.OperationalError as e:
        logger.error(f"Ошибка SQLite при получении: {e}")
        return []
    except Exception as e:
        logger.error(f"Ошибка получения прогнозов: {e}")
        return []


def get_forecast_by_id(forecast_id: int) -> dict:
    """Получение прогноза по ID"""
    ensure_db()
    
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM saved_forecasts WHERE id = ?",
            (forecast_id,)
        )
        result = cursor.fetchone()
        
        conn.close()
        return dict(result) if result else None
        
    except Exception as e:
        logger.error(f"Ошибка получения прогноза {forecast_id}: {e}")
        return None


def get_forecasts_by_city(city: str) -> list:
    """Получение прогнозов по городу"""
    ensure_db()
    
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM saved_forecasts WHERE city = ? ORDER BY updated_at DESC",
            (city,)
        )
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"Ошибка получения прогнозов для {city}: {e}")
        return []


def delete_forecast(forecast_id: int) -> bool:
    """Удаление прогноза"""
    ensure_db()
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM saved_forecasts WHERE id = ?",
            (forecast_id,)
        )
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        if deleted:
            logger.info(f"Прогноз {forecast_id} удален")
        
        return deleted
        
    except Exception as e:
        logger.error(f"Ошибка удаления прогноза {forecast_id}: {e}")
        return False


# Инициализируем БД при импорте
try:
    init_db()
    logger.info(f"База данных готова: {DB_PATH}")
except Exception as e:
    logger.error(f"Ошибка при инициализации БД: {e}")
    # Пробуем удалить файл и создать заново
    try:
        DB_PATH.unlink(missing_ok=True)
        init_db()
        logger.info("База данных пересоздана успешно")
    except:
        logger.error("Не удалось инициализировать базу данных")
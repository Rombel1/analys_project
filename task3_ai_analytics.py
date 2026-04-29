"""
ЗАДАНИЕ 3: Генерация аналитической справки с AI-элементами
"""

import pandas as pd
import numpy as np
from datetime import datetime
from task1_monitoring import PopulationMonitor
from task2_forecasting import PopulationForecaster
from utils import save_report, get_region_from_city
import logging

logger = logging.getLogger(__name__)


class AIAnalytics:
    """AI-аналитика для автоматического формирования аналитических справок"""

    def __init__(self):
        self.monitor = PopulationMonitor()
        self.forecaster = PopulationForecaster()

    # ==================== 3.1 Краткое резюме ====================

    def generate_summary(self, city_name):
        """Формирование краткого резюме динамики населения"""
        dynamics = self.monitor.get_population_dynamics(city_name)

        if dynamics is None:
            return f"Данные по городу {city_name} не найдены."

        trend = "рост" if dynamics['cagr'] > 0 else "снижение"

        period = f"{dynamics['start_year']}-{dynamics['end_year']}"

        summary = f"""
**{city_name}** демонстрирует {trend} численности населения.

За период {period} население изменилось с 
**{dynamics['start_population']:,}** до **{dynamics['end_population']:,}** человек.

**Абсолютное изменение:** {dynamics['absolute_change']:+,} чел.
**Относительное изменение:** {dynamics['relative_change']:+.1f}%
**Среднегодовой темп роста (CAGR):** {dynamics['cagr']:+.2f}%
"""
        return summary.strip()

    # ==================== 3.2 Демографические тенденции и факторы ====================

    def analyze_trends_and_factors(self, city_name):
        """Анализ демографических тенденций и факторов влияния"""
        dynamics = self.monitor.get_population_dynamics(city_name)
        demo = self.monitor.get_demographic_indicators(city_name)

        if dynamics is None:
            return {
                'trends': ['Данные отсутствуют'],
                'factors': [],
                'cagr': 0,
                'total_change_percent': 0
            }

        trends = []
        factors = []

        cagr = dynamics['cagr']
        relative_change = dynamics['relative_change']

        # 1. Тенденция роста/снижения
        if cagr > 2:
            trends.append("Экстремально высокий рост населения - темпы роста превышают 2% в год")
            factors.append({
                'name': 'Агломерационные процессы',
                'description': 'Активный приток населения из соседних регионов, развитие транспортной инфраструктуры'
            })
        elif cagr > 1:
            trends.append("Устойчивый рост населения - стабильный положительный тренд")
            factors.append({
                'name': 'Экономическое развитие',
                'description': 'Создание новых рабочих мест, рост инвестиционной привлекательности'
            })
        elif cagr > 0:
            trends.append("Слабый положительный рост - население медленно увеличивается")
        elif cagr > -1:
            trends.append("Стабилизация - незначительное снижение населения")
            factors.append({
                'name': 'Демографический переход',
                'description': 'Естественная убыль населения, старение населения, снижение рождаемости'
            })
        else:
            trends.append("Устойчивое сокращение населения - значительный отток населения")
            factors.append({
                'name': 'Миграционный отток',
                'description': 'Отток населения в более экономически развитые регионы, отсутствие перспективных рабочих мест'
            })

        # 2. Тенденция изменения темпа
        if relative_change > 20:
            trends.append("Бум роста - население увеличилось более чем на 20% за период")
        elif relative_change > 10:
            trends.append("Активный рост - значительное увеличение населения")
        elif relative_change < -10:
            trends.append("Существенное сокращение - значительная убыль населения")

        # 3. Демографические факторы
        if demo:
            if demo['birth_rate'] > 11:
                trends.append("Высокая рождаемость - коэффициент выше среднего")
                factors.append({
                    'name': 'Демографическая политика',
                    'description': f'Рождаемость {demo["birth_rate"]}‰, выше среднего по стране, способствует естественному приросту'
                })
            elif demo['birth_rate'] < 9:
                trends.append("Низкая рождаемость - коэффициент ниже среднего")
                factors.append({
                    'name': 'Демографический кризис',
                    'description': f'Рождаемость {demo["birth_rate"]}‰, ниже среднего, требует мер поддержки'
                })

            if demo['death_rate'] > 13:
                trends.append("Высокая смертность - требует внимания системы здравоохранения")
                factors.append({
                    'name': 'Здравоохранение',
                    'description': f'Смертность {demo["death_rate"]}‰, необходимо развитие медицинской инфраструктуры'
                })

        # 4. Географический фактор
        region = get_region_from_city(city_name)
        if region == 'Москва':
            factors.append({
                'name': 'Столичный статус',
                'description': 'Центр притяжения населения, высокая экономическая активность, развитая инфраструктура'
            })
        elif region == 'Санкт-Петербург':
            factors.append({
                'name': 'Крупнейший культурный и экономический центр',
                'description': 'Второй по значимости город страны, развитая промышленность и туризм'
            })
        elif region in ['Республика Татарстан', 'Свердловская область', 'Новосибирская область']:
            factors.append({
                'name': 'Региональный центр',
                'description': 'Крупный экономический и образовательный центр федерального значения'
            })

        return {
            'trends': trends,
            'factors': factors,
            'cagr': float(cagr),
            'total_change_percent': float(relative_change)
        }

    # ==================== 3.3 Прогнозная оценка ====================

    def generate_forecast_assessment(self, city_name, horizon=10):
        """Прогнозная оценка на 5-10 лет"""
        forecast = self.forecaster.forecast_city(city_name, horizon)

        if forecast is None:
            return {
                'assessment': f"Прогноз для города {city_name} не может быть построен",
                'metrics': None,
                'current_population': None,
                'forecast_5y': None,
                'forecast_10y': None,
                'cagr': None,
                'target_year': None,
                'forecast_value': None,
                'lower_bound': None,
                'upper_bound': None,
                'growth_rate': None
            }

        metrics = self.forecaster.calculate_metrics(forecast)

        last_pop = int(forecast['last_population'])
        
        # Прогноз на 5 лет (индекс 4 для горизонта 5 лет от last_year+1)
        forecast_5y = int(forecast['predictions'][4]) if len(forecast['predictions']) > 4 else None
        
        # Прогноз на 10 лет (индекс 9 для горизонта 10 лет от last_year+1)
        forecast_10y = int(forecast['predictions'][9]) if len(forecast['predictions']) > 9 else None
        
        final_pop = int(forecast['predictions'][-1])
        total_change = final_pop - last_pop
        total_change_percent = (total_change / last_pop) * 100

        if metrics and metrics.get('mape', 100) < 5:
            quality = "Высокая точность"
            confidence = "прогноз обладает высокой надежностью и может использоваться для планирования"
        elif metrics and metrics.get('mape', 100) < 10:
            quality = "Хорошая точность"
            confidence = "прогноз достаточно надежен для принятия решений"
        else:
            quality = "Удовлетворительная точность"
            confidence = "прогноз требует осторожной интерпретации и регулярного обновления"

        # Доверительный интервал
        lower = int(forecast.get('lower_bound', [final_pop * 0.95])[-1]) if forecast.get('lower_bound') is not None else int(final_pop * 0.95)
        upper = int(forecast.get('upper_bound', [final_pop * 1.05])[-1]) if forecast.get('upper_bound') is not None else int(final_pop * 1.05)

        # Среднегодовой темп роста (CAGR)
        if last_pop > 0 and horizon > 0:
            cagr = ((final_pop / last_pop) ** (1/horizon) - 1) * 100
        else:
            cagr = 0

        mape_value = metrics.get('mape', 0) if metrics else 0
        r2_value = metrics.get('r2', 0) if metrics else 0
        mae_value = metrics.get('mae', 0) if metrics else 0

        assessment = f"""
**Прогноз на {horizon} лет (до {int(forecast['future_years'][-1])} года):**

- Ожидаемая численность населения: **{int(final_pop):,}** чел.
- Изменение за период: {total_change:+,} чел. ({total_change_percent:+.1f}%)
- Среднегодовой темп: {cagr:+.2f}%

**95% доверительный интервал:** [{int(lower):,} - {int(upper):,}] чел.

**Качество прогноза:** {quality}
- {confidence}
- MAPE: {mape_value:.1f}% (средняя абсолютная процентная ошибка)
- R²: {r2_value:.3f} (коэффициент детерминации)
- MAE: {int(mae_value):,} чел. (средняя абсолютная ошибка)
"""

        return {
            'assessment': assessment.strip(),
            'metrics': metrics,
            'current_population': last_pop,
            'forecast_5y': forecast_5y,
            'forecast_10y': forecast_10y,
            'cagr': round(cagr, 2),
            'target_year': int(forecast['future_years'][-1]),
            'forecast_value': final_pop,
            'lower_bound': lower,
            'upper_bound': upper,
            'growth_rate': round(cagr, 2)
        }

    # ==================== 3.4 Рекомендации ====================

    def generate_recommendations(self, city_name):
        """Составление рекомендаций по социальной политике и территориальному планированию"""
        dynamics = self.monitor.get_population_dynamics(city_name)
        demo = self.monitor.get_demographic_indicators(city_name)
        forecast = self.forecaster.forecast_city(city_name, 10)

        if dynamics is None:
            return []

        recommendations = []

        cagr = dynamics['cagr']

        # Рекомендации на основе динамики
        if cagr > 1:
            recommendations.append({
                'category': 'ИНФРАСТРУКТУРНОЕ РАЗВИТИЕ',
                'title': 'Опережающее развитие социальной инфраструктуры',
                'description': 'В связи с активным ростом населения необходимо опережающее развитие инфраструктуры',
                'priority': 'high',
                'actions': [
                    'Планирование строительства новых школ и детских садов (+2-3 в год)',
                    'Расширение сети поликлиник и медицинских учреждений',
                    'Развитие транспортной инфраструктуры (метро, МЦД, дороги)',
                    'Увеличение объемов жилищного строительства (+10-15% в год)',
                    'Создание новых парков и зон отдыха'
                ]
            })
            recommendations.append({
                'category': 'СОЦИАЛЬНАЯ ПОЛИТИКА',
                'title': 'Поддержка семей с детьми',
                'description': 'Для сохранения положительной динамики роста населения',
                'priority': 'medium',
                'actions': [
                    'Материальная поддержка молодых семей',
                    'Льготная ипотека для семей с детьми',
                    'Развитие системы дополнительного образования'
                ]
            })
        elif cagr < 0:
            recommendations.append({
                'category': 'ДЕМОГРАФИЧЕСКАЯ ПОЛИТИКА',
                'title': 'Стимулирование рождаемости и удержание населения',
                'description': 'Для противодействия снижению численности населения',
                'priority': 'high',
                'actions': [
                    'Внедрение программ поддержки семей с детьми (выплаты, сертификаты)',
                    'Создание новых рабочих мест (технопарки, промзоны)',
                    'Развитие системы профессионального образования',
                    'Повышение качества городской среды (реновация, благоустройство)',
                    'Программы по привлечению квалифицированных специалистов'
                ]
            })

        # Рекомендации на основе демографии
        if demo:
            if demo['birth_rate'] < 9:
                recommendations.append({
                    'category': 'ПОДДЕРЖКА РОЖДАЕМОСТИ',
                    'title': 'Повышение рождаемости',
                    'description': f'Текущий уровень рождаемости {demo["birth_rate"]}‰ ниже среднего',
                    'priority': 'high',
                    'actions': [
                        'Введение дополнительных выплат при рождении детей',
                        'Развитие сети детских дошкольных учреждений',
                        'Программы доступного жилья для семей с детьми',
                        'Налоговые льготы для многодетных семей'
                    ]
                })

            if demo['death_rate'] > 13:
                recommendations.append({
                    'category': 'ЗДРАВООХРАНЕНИЕ',
                    'title': 'Снижение смертности',
                    'description': f'Текущий уровень смертности {demo["death_rate"]}‰ выше среднего',
                    'priority': 'high',
                    'actions': [
                        'Модернизация системы здравоохранения',
                        'Профилактика сердечно-сосудистых заболеваний',
                        'Развитие гериатрической помощи',
                        'Привлечение квалифицированных врачей'
                    ]
                })

        # Рекомендации на основе прогноза
        if forecast:
            final_change = (forecast['predictions'][-1] / forecast['last_population'] - 1) * 100
            if final_change > 10:
                recommendations.append({
                    'category': 'СТРАТЕГИЧЕСКОЕ ПЛАНИРОВАНИЕ',
                    'title': 'Долгосрочное развитие города',
                    'description': 'Ожидается значительный рост населения',
                    'priority': 'high',
                    'actions': [
                        'Разработка мастер-плана развития города на 15-20 лет',
                        'Опережающее строительство жилья и инфраструктуры',
                        'Развитие системы общественного транспорта',
                        'Создание новых общественных пространств'
                    ]
                })
            elif final_change < -5:
                recommendations.append({
                    'category': 'АНТИКРИЗИСНЫЕ МЕРЫ',
                    'title': 'Предотвращение депопуляции',
                    'description': 'Прогнозируется значительное снижение населения',
                    'priority': 'critical',
                    'actions': [
                        'Разработка комплексной программы возрождения территорий',
                        'Стимулирование малого и среднего бизнеса',
                        'Повышение инвестиционной привлекательности',
                        'Создание особой экономической зоны'
                    ]
                })
            else:
                recommendations.append({
                    'category': 'РАЗВИТИЕ',
                    'title': 'Сбалансированное развитие',
                    'description': 'Ожидается умеренное изменение численности населения',
                    'priority': 'medium',
                    'actions': [
                        'Поддержание текущего уровня инфраструктуры',
                        'Стимулирование экономической активности',
                        'Улучшение качества городской среды'
                    ]
                })

        # Общая рекомендация (всегда добавляем)
        recommendations.append({
            'category': 'МОНИТОРИНГ',
            'title': 'Регулярный анализ демографической ситуации',
            'description': 'Для своевременного реагирования на изменения',
            'priority': 'medium',
            'actions': [
                'Ежегодный пересмотр прогнозов',
                'Корректировка стратегий развития с учетом актуальных данных',
                'Создание демографического совета при администрации'
            ]
        })

        return recommendations

    # ==================== Полный отчет ====================

    def generate_full_report(self, city_name, forecast_horizon=10):
        """Генерация полной аналитической справки с разделением на пункты"""
        logger.info(f"Генерация аналитической справки для города: {city_name}")

        # 3.1 Краткое резюме
        summary = self.generate_summary(city_name)

        # 3.2 Демографические тенденции и факторы
        trends_factors = self.analyze_trends_and_factors(city_name)

        # 3.3 Прогнозная оценка
        forecast_assessment = self.generate_forecast_assessment(city_name, forecast_horizon)

        # 3.4 Рекомендации
        recommendations = self.generate_recommendations(city_name)

        # Заключение
        dynamics = self.monitor.get_population_dynamics(city_name)
        conclusion = self._generate_conclusion(dynamics, forecast_assessment)

        report = {
            'city': city_name,
            'region': get_region_from_city(city_name),
            'generated_at': datetime.now().isoformat(),

            # Пункт 3.1
            'section_31_summary': summary,

            # Пункт 3.2
            'section_32_trends_and_factors': {
                'trends': trends_factors.get('trends', []),
                'factors': trends_factors.get('factors', []),
                'cagr': trends_factors.get('cagr', 0)
            },

            # Пункт 3.3
            'section_33_forecast': forecast_assessment,

            # Пункт 3.4
            'section_34_recommendations': recommendations,

            # Заключение
            'section_35_conclusion': conclusion,

            # Данные для графика
            'data': {
                'historical': dynamics,
                'forecast': self.forecaster.forecast_city(city_name, forecast_horizon)
            }
        }

        return report

    def _generate_conclusion(self, dynamics, forecast_assessment):
        """Генерация заключения"""
        if dynamics is None:
            return "Данные для заключения отсутствуют."

        conclusion = f"""
**Заключение по демографической ситуации в городе {dynamics['city']}**

Анализ демографической ситуации показывает, что город находится в фазе 
**{'роста' if dynamics['cagr'] > 0 else 'стабилизации/сокращения'}** численности населения.

**Ключевые выводы:**
1. {'Положительная' if dynamics['cagr'] > 0 else 'Отрицательная'} динамика населения: 
   {dynamics['relative_change']:+.1f}% за период {dynamics['start_year']}-{dynamics['end_year']}
2. Среднегодовой темп роста: {dynamics['cagr']:+.2f}%
"""

        if forecast_assessment and forecast_assessment.get('target_year'):
            conclusion += f"""
3. Прогноз на {forecast_assessment['target_year']} год: {forecast_assessment.get('forecast_value', 'Н/Д'):,} чел.
4. Качество прогноза: MAPE = {forecast_assessment.get('metrics', {}).get('mape', 'N/A')}%

**Приоритетные направления развития:**
- Реализация предложенных мер в соответствии с их приоритетностью
- Регулярный мониторинг демографических показателей
- Корректировка стратегий с учетом актуальных данных
"""

        return conclusion.strip()

    def generate_markdown_report(self, city_name, forecast_horizon=10):
        """Генерация справки в формате Markdown для экспорта"""
        report = self.generate_full_report(city_name, forecast_horizon)

        if report is None:
            return None

        md = f"""# Аналитическая справка: {report['city']}

**Регион:** {report['region']}  
**Дата генерации:** {report['generated_at'][:19]}

---

## 3.1 Краткое резюме динамики населения

{report['section_31_summary']}

---

## 3.2 Демографические тенденции и возможные факторы влияния

### Выявленные тенденции:
{chr(10).join(['- ' + t for t in report['section_32_trends_and_factors']['trends']])}

### Факторы влияния:
{chr(10).join([f"**{f['name']}:** {f['description']}" for f in report['section_32_trends_and_factors']['factors']])}

---

## 3.3 Прогнозная оценка на 5-10 лет

{report['section_33_forecast']['assessment']}

---

## 3.4 Рекомендации по социальной политике и территориальному планированию

"""
        for rec in report['section_34_recommendations']:
            priority_text = "КРИТИЧЕСКИЙ" if rec['priority'] == 'critical' else "ВЫСОКИЙ" if rec['priority'] == 'high' else "СРЕДНИЙ"
            md += f"""
### {rec['title']}

**Категория:** {rec['category']}  
**Приоритет:** {priority_text}

**Рекомендуемые действия:**
{chr(10).join(['- ' + a for a in rec['actions']])}

"""

        md += f"""

---

## 3.5 Заключение

{report['section_35_conclusion']}

---

*Отчет сгенерирован автоматически аналитической системой демографии РФ*
"""
        return md

// Конфигурация API
const API_BASE = window.location.origin;

// Состояние приложения
let state = {
    regions: [],
    cities: [],
    selectedCity: '',
    savedForecasts: []
};

// DOM элементы
const regionFilter = document.getElementById('regionFilter');
const cityInput = document.getElementById('cityInput');
const cityDropdown = document.getElementById('cityDropdown');
const horizonSelect = document.getElementById('horizonSelect');
const btnForecast = document.getElementById('btnForecast');
const btnAnalytics = document.getElementById('btnAnalytics');
const btnHeatmap = document.getElementById('btnHeatmap');
const btnTopChanges = document.getElementById('btnTopChanges');
const btnSavedForecasts = document.getElementById('btnSavedForecasts');
const btnClear = document.getElementById('btnClear');
const outputArea = document.getElementById('outputArea');

// Инициализация
async function init() {
    await loadRegions();
    await loadCities();
    await loadSavedForecasts();
    setupEventListeners();
}

// Загрузка регионов
async function loadRegions() {
    try {
        const response = await fetch(`${API_BASE}/api/regions`);
        state.regions = await response.json();
        state.regions.forEach(region => {
            const option = document.createElement('option');
            option.value = region;
            option.textContent = region;
            regionFilter.appendChild(option);
        });
    } catch (error) {
        console.error('Ошибка загрузки регионов:', error);
    }
}

// Загрузка городов
async function loadCities(region = '') {
    try {
        const url = region 
            ? `${API_BASE}/api/cities?region=${encodeURIComponent(region)}`
            : `${API_BASE}/api/cities`;
        const response = await fetch(url);
        state.cities = await response.json();
    } catch (error) {
        console.error('Ошибка загрузки городов:', error);
    }
}

// Загрузка сохраненных прогнозов
async function loadSavedForecasts() {
    try {
        const response = await fetch(`${API_BASE}/api/forecasts/saved`);
        if (response.ok) {
            state.savedForecasts = await response.json();
        }
    } catch (error) {
        console.error('Ошибка загрузки сохраненных прогнозов:', error);
    }
}

// Показ подсказок для города
function showCitySuggestions(query = '') {
    if (!state.cities.length) {
        cityDropdown.classList.remove('show');
        return;
    }

    const filtered = query
        ? state.cities.filter(c => c.name.toLowerCase().includes(query.toLowerCase()))
        : state.cities.slice(0, 10);

    if (filtered.length === 0) {
        cityDropdown.innerHTML = '<div class="suggestion-row" style="cursor:default;color:#a0aec0;">Ничего не найдено</div>';
    } else {
        cityDropdown.innerHTML = filtered.slice(0, 15).map(city =>
            `<div class="suggestion-row" onclick="selectCity('${escapeHtml(city.name)}')">
                <span>${city.name}</span>
                <span class="suggestion-pop">${city.population.toLocaleString()} чел.</span>
            </div>`
        ).join('');
    }
    cityDropdown.classList.add('show');
}

// Выбор города
window.selectCity = function(cityName) {
    state.selectedCity = cityName;
    cityInput.value = cityName;
    cityDropdown.classList.remove('show');
    btnForecast.disabled = false;
    btnAnalytics.disabled = false;
};

// Экранирование HTML
function escapeHtml(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

// Настройка обработчиков событий
function setupEventListeners() {
    regionFilter.addEventListener('change', async () => {
        await loadCities(regionFilter.value);
        cityInput.value = '';
        state.selectedCity = '';
        btnForecast.disabled = true;
        btnAnalytics.disabled = true;
    });

    cityInput.addEventListener('input', () => {
        state.selectedCity = '';
        btnForecast.disabled = true;
        btnAnalytics.disabled = true;
        showCitySuggestions(cityInput.value);
    });

    cityInput.addEventListener('focus', () => {
        showCitySuggestions(cityInput.value);
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('#cityInput') && !e.target.closest('#cityDropdown')) {
            cityDropdown.classList.remove('show');
        }
    });

    btnForecast.addEventListener('click', getForecast);
    btnAnalytics.addEventListener('click', getAnalytics);
    btnHeatmap.addEventListener('click', getHeatmap);
    btnTopChanges.addEventListener('click', getTopChanges);
    btnSavedForecasts.addEventListener('click', showSavedForecasts);
    btnClear.addEventListener('click', clearOutput);
}

// Получение прогноза
async function getForecast() {
    if (!state.selectedCity) return;
    showLoading('Загрузка прогноза...');
    
    try {
        const horizon = horizonSelect.value;
        const response = await fetch(
            `${API_BASE}/api/forecast/${encodeURIComponent(state.selectedCity)}?horizon=${horizon}&include_metrics=true`
        );
        
        if (!response.ok) throw new Error(`Ошибка ${response.status}`);
        const data = await response.json();
        renderForecast(data);
        await saveForecastToDb(data);
    } catch (error) {
        showError(`Ошибка загрузки прогноза: ${error.message}`);
    }
}

// Сохранение прогноза в БД
async function saveForecastToDb(data) {
    try {
        await fetch(`${API_BASE}/api/forecasts/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                city: data.city,
                horizon: data.horizon,
                forecast_data: data
            })
        });
        await loadSavedForecasts();
    } catch (error) {
        console.error('Ошибка сохранения прогноза:', error);
    }
}

// Получение аналитики
async function getAnalytics() {
    if (!state.selectedCity) return;
    showLoading('Генерация аналитической справки...');
    
    try {
        const response = await fetch(`${API_BASE}/api/ai/report/${encodeURIComponent(state.selectedCity)}`);
        if (!response.ok) throw new Error(`Ошибка ${response.status}`);
        const data = await response.json();
        renderAnalytics(data);
    } catch (error) {
        showError(`Ошибка загрузки аналитики: ${error.message}`);
    }
}

// Тепловая карта
async function getHeatmap() {
    showLoading('Загрузка тепловой карты...');
    
    try {
        const response = await fetch(`${API_BASE}/api/regional_population`);
        if (!response.ok) throw new Error(`Ошибка ${response.status}`);
        const data = await response.json();
        renderHeatmap(data);
    } catch (error) {
        showError(`Ошибка загрузки: ${error.message}`);
    }
}

// Топ изменений
async function getTopChanges() {
    showLoading('Загрузка данных...');
    
    try {
        const response = await fetch(`${API_BASE}/api/top_changes?n=10&min_population=500000`);
        if (!response.ok) throw new Error(`Ошибка ${response.status}`);
        const data = await response.json();
        renderTopChanges(data);
    } catch (error) {
        showError(`Ошибка загрузки: ${error.message}`);
    }
}

// Показ сохраненных прогнозов
async function showSavedForecasts() {
    await loadSavedForecasts();
    
    if (!state.savedForecasts.length) {
        outputArea.innerHTML = '<div class="panel"><div class="panel-body"><p>Нет сохраненных прогнозов</p></div></div>';
        outputArea.style.display = 'block';
        return;
    }

    const html = `
        <div class="panel">
            <div class="panel-header">Сохраненные прогнозы</div>
            <div class="panel-body">
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Город</th>
                                <th>Горизонт</th>
                                <th>Текущее население</th>
                                <th>Прогноз (конечный)</th>
                                <th>Дата сохранения</th>
                                <th>Действия</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${state.savedForecasts.map(f => {
                                const data = JSON.parse(f.forecast_data);
                                return `
                                    <tr>
                                        <td>${f.city}</td>
                                        <td>${f.horizon} лет</td>
                                        <td>${data.last_population.toLocaleString()}</td>
                                        <td>${Math.round(data.predictions[data.predictions.length-1]).toLocaleString()}</td>
                                        <td>${new Date(f.created_at).toLocaleString('ru-RU')}</td>
                                        <td>
                                            <button class="btn btn-outline" onclick="loadSavedForecast(${f.id})" style="padding:4px 8px;font-size:12px;">Загрузить</button>
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    outputArea.innerHTML = html;
    outputArea.style.display = 'block';
}

// Загрузка сохраненного прогноза
window.loadSavedForecast = async function(id) {
    try {
        const response = await fetch(`${API_BASE}/api/forecasts/saved/${id}`);
        if (!response.ok) throw new Error('Не найден');
        const saved = await response.json();
        const data = JSON.parse(saved.forecast_data);
        state.selectedCity = data.city;
        cityInput.value = data.city;
        btnForecast.disabled = false;
        btnAnalytics.disabled = false;
        renderForecast(data);
    } catch (error) {
        showError('Ошибка загрузки сохраненного прогноза');
    }
};

// Отрисовка прогноза
function renderForecast(data) {
    const html = `
        <div class="panel">
            <div class="panel-header">Прогноз численности населения: ${data.city}</div>
            <div class="panel-body">
                <div class="info-banner">
                    <strong>Текущая численность:</strong> ${data.last_population.toLocaleString()} чел. (${data.last_year} год)<br>
                    <strong>Горизонт прогнозирования:</strong> ${data.horizon} лет
                </div>
                ${data.metrics ? `
                <div class="metrics-row">
                    <div class="metric-box">
                        <div class="metric-number">${data.metrics.mape.toFixed(2)}%</div>
                        <div class="metric-caption">MAPE</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-number">${data.metrics.rmse.toLocaleString()}</div>
                        <div class="metric-caption">RMSE</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-number">${data.metrics.mae.toLocaleString()}</div>
                        <div class="metric-caption">MAE</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-number">${data.metrics.r2.toFixed(4)}</div>
                        <div class="metric-caption">R-squared</div>
                    </div>
                </div>
                <div class="info-banner">
                    <strong>Качество модели:</strong> ${data.metrics.interpretation}
                </div>` : ''}
                <div class="chart-container">
                    <canvas id="forecastChart" height="300"></canvas>
                </div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Год</th>
                                <th style="text-align:right;">Прогноз (чел.)</th>
                                <th style="text-align:right;">Нижняя граница</th>
                                <th style="text-align:right;">Верхняя граница</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.future_years.map((year, i) => `
                                <tr>
                                    <td style="text-align:center;">${year}</td>
                                    <td style="text-align:right;">${Math.round(data.predictions[i]).toLocaleString()}</td>
                                    <td style="text-align:right;">${Math.round(data.lower_bound[i]).toLocaleString()}</td>
                                    <td style="text-align:right;">${Math.round(data.upper_bound[i]).toLocaleString()}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    outputArea.innerHTML = html;
    outputArea.style.display = 'block';
    outputArea.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => createForecastChart(data), 100);
}

// Создание графика
function createForecastChart(data) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.future_years,
            datasets: [
                {
                    label: 'Прогноз',
                    data: data.predictions,
                    borderColor: '#4a5568',
                    backgroundColor: 'rgba(74,85,104,0.08)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3
                },
                {
                    label: 'Нижняя граница',
                    data: data.lower_bound,
                    borderColor: '#a0aec0',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Верхняя граница',
                    data: data.upper_bound,
                    borderColor: '#a0aec0',
                    borderDash: [5, 5],
                    fill: '-1',
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${Math.round(ctx.parsed.y).toLocaleString()} чел.`
                    }
                }
            },
            scales: {
                y: {
                    ticks: { callback: v => v.toLocaleString() }
                }
            }
        }
    });
}

// Отрисовка аналитики
function renderAnalytics(data) {
    const html = `
        <div class="panel">
            <div class="panel-header">Аналитическая справка: ${data.city}</div>
            <div class="panel-body">
                <div class="info-banner">
                    <strong>Регион:</strong> ${data.region || 'Н/Д'}<br>
                    <strong>Дата:</strong> ${data.generated_at ? new Date(data.generated_at).toLocaleString('ru-RU') : 'Н/Д'}
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">Краткое резюме</div>
            <div class="panel-body">
                <div class="info-banner">${(data.section_31_summary || 'Нет данных').replace(/\n/g, '<br>')}</div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">Тенденции и факторы</div>
            <div class="panel-body">
                ${data.section_32_trends_and_factors ? `
                    <h4 style="margin-bottom:10px;color:#4a5568;">Выявленные тенденции:</h4>
                    <ul style="list-style:none;padding:0;">
                        ${(data.section_32_trends_and_factors.trends || []).map(t => 
                            `<li style="padding:8px 12px;margin:4px 0;background:#f7fafc;border-left:3px solid #4a5568;border-radius:2px;">${t}</li>`
                        ).join('')}
                    </ul>
                    <h4 style="margin:16px 0 10px;color:#4a5568;">Факторы влияния:</h4>
                    <ul style="list-style:none;padding:0;">
                        ${(data.section_32_trends_and_factors.factors || []).map(f => 
                            `<li style="padding:8px 12px;margin:4px 0;background:#f7fafc;border-left:3px solid #a0aec0;border-radius:2px;">
                                <strong>${f.name}:</strong> ${f.description}
                            </li>`
                        ).join('')}
                    </ul>
                ` : '<p>Нет данных</p>'}
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">Прогнозная оценка</div>
            <div class="panel-body">
                ${data.section_33_forecast ? `
                    <div class="info-banner">
                        <strong>Текущая численность:</strong> ${data.section_33_forecast.current_population?.toLocaleString() || 'Н/Д'} чел.<br>
                        <strong>Прогноз на 5 лет:</strong> ${data.section_33_forecast.forecast_5y?.toLocaleString() || 'Н/Д'} чел.<br>
                        <strong>Прогноз на 10 лет:</strong> ${data.section_33_forecast.forecast_10y?.toLocaleString() || 'Н/Д'} чел.<br>
                        <strong>CAGR:</strong> ${data.section_33_forecast.cagr != null ? data.section_33_forecast.cagr.toFixed(2) + '%' : 'Н/Д'}
                    </div>
                ` : '<p>Данные отсутствуют</p>'}
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">Рекомендации</div>
            <div class="panel-body">
                ${data.section_34_recommendations && data.section_34_recommendations.length > 0 
                    ? data.section_34_recommendations.map(rec => {
                        const tagClass = {critical:'tag-critical',high:'tag-high',medium:'tag-medium',low:'tag-low'}[rec.priority] || 'tag-medium';
                        const tagLabel = {critical:'КРИТИЧЕСКИЙ',high:'ВЫСОКИЙ',medium:'СРЕДНИЙ',low:'НИЗКИЙ'}[rec.priority] || 'СРЕДНИЙ';
                        return `
                        <div class="recommendation-item">
                            <span class="priority-tag ${tagClass}">${tagLabel}</span>
                            <h4 style="margin:6px 0;color:#2d3748;">${rec.title || 'Рекомендация'}</h4>
                            <p style="color:#718096;font-size:13px;">${rec.category || ''}</p>
                            ${rec.description ? `<p style="margin:6px 0;">${rec.description}</p>` : ''}
                        </div>`;
                    }).join('')
                    : '<p>Рекомендации отсутствуют</p>'}
            </div>
        </div>
        
        ${data.section_35_conclusion ? `
        <div class="panel">
            <div class="panel-header">Заключение</div>
            <div class="panel-body">
                <div class="info-banner">${data.section_35_conclusion.replace(/\n/g, '<br>')}</div>
            </div>
        </div>` : ''}
        
        <div style="text-align:center;margin:16px 0;">
            <button class="btn btn-outline" onclick="window.open('/api/ai/report/${encodeURIComponent(data.city)}/markdown', '_blank')">
                Скачать Markdown
            </button>
        </div>
    `;

    outputArea.innerHTML = html;
    outputArea.style.display = 'block';
    outputArea.scrollIntoView({ behavior: 'smooth' });
}

// Отрисовка тепловой карты
function renderHeatmap(data) {
    if (!data || !data.length) {
        showError('Нет данных');
        return;
    }
    
    const html = `
        <div class="panel">
            <div class="panel-header">Тепловая карта регионов (Топ-15)</div>
            <div class="panel-body">
                <div class="chart-container">
                    <div id="heatmapPlot" style="height:400px;"></div>
                </div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Регион</th>
                                <th style="text-align:right;">Население</th>
                                <th style="text-align:right;">Доля</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${(() => {
                                const total = data.reduce((s, r) => s + r.population, 0);
                                return data.map(d => `
                                    <tr>
                                        <td>${d.region}</td>
                                        <td style="text-align:right;">${d.population.toLocaleString()}</td>
                                        <td style="text-align:right;">${((d.population/total)*100).toFixed(2)}%</td>
                                    </tr>
                                `).join('');
                            })()}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    outputArea.innerHTML = html;
    outputArea.style.display = 'block';
    outputArea.scrollIntoView({ behavior: 'smooth' });
    
    setTimeout(() => {
        const regions = data.slice(0, 15).map(d => d.region);
        const populations = data.slice(0, 15).map(d => d.population);
        const maxPop = Math.max(...populations);
        
        Plotly.newPlot('heatmapPlot', [{
            x: regions,
            y: populations,
            type: 'bar',
            marker: {
                color: populations.map(v => `rgba(74,85,104,${0.3 + v/maxPop * 0.7})`)
            }
        }], {
            margin: { t: 10, b: 100 },
            xaxis: { tickangle: -45 }
        });
    }, 100);
}

// Отрисовка топа изменений
function renderTopChanges(data) {
    const html = `
        <div class="panel">
            <div class="panel-header">Топ-10 городов по изменению населения</div>
            <div class="panel-body">
                <div class="split-grid">
                    <div class="result-card">
                        <div class="result-title trend-positive">Растущие города</div>
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Город</th>
                                    <th style="text-align:right;">Изменение</th>
                                    <th style="text-align:right;">CAGR</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.growing.map(c => `
                                    <tr>
                                        <td>${c.city}</td>
                                        <td style="text-align:right;color:#38a169;">+${c.relative_change.toFixed(1)}%</td>
                                        <td style="text-align:right;">${c.cagr.toFixed(2)}%</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                    <div class="result-card">
                        <div class="result-title trend-negative">Убывающие города</div>
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Город</th>
                                    <th style="text-align:right;">Изменение</th>
                                    <th style="text-align:right;">CAGR</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.declining.map(c => `
                                    <tr>
                                        <td>${c.city}</td>
                                        <td style="text-align:right;color:#e53e3e;">${c.relative_change.toFixed(1)}%</td>
                                        <td style="text-align:right;">${c.cagr.toFixed(2)}%</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    outputArea.innerHTML = html;
    outputArea.style.display = 'block';
    outputArea.scrollIntoView({ behavior: 'smooth' });
}

// Вспомогательные функции
function showLoading(message) {
    outputArea.innerHTML = `
        <div class="panel">
            <div class="panel-body">
                <div class="loading-box">
                    <div class="spinner-icon"></div>
                    ${message}
                </div>
            </div>
        </div>
    `;
    outputArea.style.display = 'block';
    outputArea.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    outputArea.innerHTML = `
        <div class="panel">
            <div class="panel-body">
                <div class="error-banner">${message}</div>
            </div>
        </div>
    `;
    outputArea.style.display = 'block';
    outputArea.scrollIntoView({ behavior: 'smooth' });
}

function clearOutput() {
    outputArea.style.display = 'none';
    outputArea.innerHTML = '';
}

// Запуск приложения
init();
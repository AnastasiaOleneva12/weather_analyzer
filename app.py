import streamlit as st
import pandas as pd
import matplotlib.dates as mdates
import requests
import plotly.express as px
import time
from concurrent.futures import ThreadPoolExecutor
import aiohttp

st.title('Анализ погоды и температурных аномалий')

uploaded_file = st.file_uploader('Загрузите CSV-файл с историческими данными', type=['csv'])

city = st.selectbox('Выберите город', ['New York', 'London', 'Paris', 'Tokyo', 'Moscow', 'Sydney',
                                       'Berlin', 'Beijing', 'Rio de Janeiro', 'Dubai', 'Los Angeles',
                                       'Singapore', 'Mumbai', 'Cairo', 'Mexico City'])

api_key = st.text_input('Введите API-ключ OpemWeatherMap', type='password')

st.markdown('## *Работа с историческими данными*')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    city_df = df[df['city'] == city]
    #Скользящее среднее

    def calculate_rolling_mean(data):
        roll_mean_df = data.copy()
        roll_mean_df = roll_mean_df.set_index('timestamp')

        roll_mean_df['rolling_mean'] = roll_mean_df['temperature'].rolling(window=30).mean()
        return roll_mean_df.dropna()


    #Средняя температура и стандартное отклонение по сезонам

    def calculate_seasons_stats(data):
        seasons_df = data.copy()
        return seasons_df.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()


    #Температурные аномалии

    def search_of_anomalies(data, seasons_data):
        anomalies_df = data.copy()
        anomalies_df = anomalies_df.merge(seasons_data, how='left')

        anomalies_df['is_anomaly'] = 0
        anomalies_df.loc[(anomalies_df['temperature'] >= anomalies_df['mean'] + 2 * anomalies_df['std']) | (anomalies_df['temperature'] <= anomalies_df['mean'] - 2 * anomalies_df['std']), 'is_anomaly'] = 1
        return anomalies_df


    #Стандартное выполнение
    start_time = time.time()
    roll_mean_sync = calculate_rolling_mean(df)
    seasons_df_sync = calculate_seasons_stats(df)
    anomalies_sync = search_of_anomalies(df, seasons_df_sync)
    sync_time = time.time() - start_time

    #Параллельное выполнение
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        roll_mean_future = executor.submit(calculate_rolling_mean, df)
        seasons_future = executor.submit(calculate_seasons_stats, df)
        roll_mean_parallel = roll_mean_future.result()
        seasons_parallel = seasons_future.result()
        anomalies_parallel = search_of_anomalies(df, seasons_parallel)
    parallel_time = time.time() - start_time


    roll_mean_df = calculate_rolling_mean(city_df)
    st.subheader('Скользящее среднее:')
    st.dataframe(roll_mean_df)

    fig1 = px.line(roll_mean_df, x = roll_mean_df.index, y=['temperature', 'rolling_mean'],
                   title='Скользящее среднее температуры')
    st.plotly_chart(fig1)


    seasons_df = calculate_seasons_stats(city_df)
    st.subheader('Средняя температура и стандартное отклонение по сезонам:')
    st.dataframe(seasons_df)

    fig2 = px.bar(seasons_df, x='season', y='mean', error_y='std',
                  title='Средняя температура и стандартное отклонение по сезонам')
    st.plotly_chart(fig2)


    anomalies_df = search_of_anomalies(city_df, seasons_df)
    st.subheader('Температурные аномалии')
    st.dataframe(anomalies_df[anomalies_df['is_anomaly'] == 1])

    season = st.selectbox('Выберите сезон', anomalies_df['season'].unique())
    filtered_df = anomalies_df[anomalies_df['season'] == season]
    fig3 = px.scatter(filtered_df, x='timestamp', y='temperature', color='is_anomaly',
                      color_discrete_map={False: 'blue', True: 'red'},
                      title=f'Температура за сезон {season}')
    fig3.update_traces(marker=dict(size=7, opacity=0.7))
    fig3.update_layout(legend_title='Аномалия')
    st.plotly_chart(fig3)


    st.subheader('Сравнение времени выполнения:')
    st.write(f'Синхронное выполненение: {round(sync_time, 2)} секунд')
    st.write(f'Параллельное выполнение выполненение: {round(parallel_time, 2)} секунд')
    st.write('При небольшом количестве данных разница в выполнении незначительная')


    #Мониторинг текущей температуры
    #Текущая температура в выбранном городе
    st.markdown('## *Работа с текущей температурой*')
    def get_temperature(city, api_key):
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}'
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temperature = data['main']['temp']
            return temperature
        elif response.status_code == 401:
            return 'Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.'
        else:
            return None

    if st.button('Узнать текущую температуру'):
        if api_key:
            temperature = get_temperature(city, api_key)
            if isinstance(temperature, str):
                st.error(temperature)
            else:
                st.success(f'Текущая температура в {city}: {temperature}')

    curr_temp_df = df.copy()
    curr_temp_df = curr_temp_df.loc[:, ['city', 'season', 'temperature']]
    curr_temp_df = curr_temp_df[curr_temp_df['season'] == 'spring']
    curr_temp_df = curr_temp_df.groupby('city')['temperature'].agg(['mean', 'std']).reset_index()
    curr_temp_df['current_temperature'] = curr_temp_df['city'].apply(lambda x: get_temperature(x, api_key))
    curr_temp_df['is_anomaly'] = 'no'
    curr_temp_df.loc[(curr_temp_df['current_temperature'] >= curr_temp_df['mean'] + 2 * curr_temp_df['std']) | (curr_temp_df['current_temperature'] <= curr_temp_df['mean'] - 2 * curr_temp_df['std']), 'is_anomaly'] = 'yes'

    st.subheader('Сравненение средней и текущей температур')
    st.write('Рассмотрим среднюю температуру и стандартное отклонение в текущем сезоне (весна), '
             'проверим, являются ли текущие значения температуры аномальными или нет')
    st.dataframe(curr_temp_df)

        #Асинхронное получение текущей температуры

    async def get_temperature_async(city, api_key):
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}'

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                if response.status == 200:
                    return data['main']['temp']
                elif response.status == 401:
                    return 'Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.'
                else:
                    return None

    cities = df['city'].unique()

    def get_temps_sync(cities, api_key):
        return [get_temperature(city, api_key) for city in cities]

    async def get_temps_async(cities, api_key):
        return [get_temperature_async(city, api_key) for city in cities]

    start_sync = time.time()
    sync_temps = get_temps_sync(cities, api_key)
    end_sync = time.time()
    time_sync = round(end_sync - start_sync, 2)

    start_async = time.time()
    async_temps = get_temps_async(cities, api_key)
    end_async = time.time()
    time_async = round(end_async - start_async, 2)

    st.subheader('Сравнение времени синхронного и асинхронного получения текущей температуры')
    st.write(f'Время синхронной работы: {time_sync} сек')
    st.write(f'Время асинхронной работы: {time_async} сек')
    st.write('Время синхронной работы значительно превышает время асинхронной. Соответственно использовать асинхронные методы предпочтительнее.')


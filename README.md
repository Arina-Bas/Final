### ОТЧЕТ
### Практический кейс  «Прогнозирование размеров сварного шва при электронно-лучевой сварке тонкостенных конструкций аэрокосмического назначения» 

В данном отчете представлен процесс анализа и моделирования данных, используя библиотеку Pandas для обработки данных и библиотеки Scikit-learn, Keras и XGBoost для построения моделей. 

### Задача: 
Решить задачу регрессии (одним или несколькими методами)  для предсказания глубины и ширины сварного соединения.

### Импорт библиотек
Начинаем с импорта необходимых библиотек:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

### Загрузка и подготовка данных
Данные загружаются из CSV файла, после чего устанавливаются опции отображения для полной видимости всех колонок и строк:
df = pd.read_csv('ebw_data.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.info())
Удаляем дубликаты, если они имеются в данных:
df = df.drop_duplicates()
df_c = df
df_c.to_csv('cleaned_df_final_work.csv')

### Удаление выбросов
Функция remove_outliers применяется для обработки колонок, заданных в списке columns_for_remove_outliers, с целью удаления выбросов на основе межквартильного размаха (IQR):
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

columns_for_remove_outliers = ['IW', 'IF', 'VW', 'FP', 'Depth', 'Width']
for column in columns_for_remove_outliers:
    df_c = remove_outliers(df_c, column)

### Анализ распределения данных
Используем Q-Q график для проверки нормальности распределения данных:
columns_for_tests_and_visualisation = ['IW', 'IF', 'VW', 'FP', 'Depth', 'Width']
for column in columns_for_tests_and_visualisation:
    plt.figure()
    st.probplot(df_c[column], dist='norm', plot=plt)
    plt.title(f'Q-Q график для колонки {column}')
    plt.savefig(f'Q-Q график для колонки {column}.png')
Также создается тепловая карта для визуализации корреляции между переменными:
plt.figure(figsize=(16, 14))
corr_matrix = df_c[columns_for_tests_and_visualisation].corr()
sb.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig('Тепловая карта.png')
![image](https://github.com/user-attachments/assets/27a49278-7c9b-464d-a8ba-957e060a969c)


### Проверка нормальности распределения с помощью теста Андерсона
Проводим тест Андерсона для проверки нормальности распределения для колонок Depth и Width:
result = st.anderson(df_c['Depth'])
result = st.anderson(df_c['Width'])
# Аналогичные действия для колонки Width

### Построение модели линейной регрессии
Данные делятся на обучающую и тестовую выборки:
train_size = int(len(df) * 0.8)
train, test = df_c.iloc[:train_size], df_c.iloc[train_size:]
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Depth']
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
X_test = test[['IW', 'IF', 'VW', 'FP']]
y_test = test['Depth']
y_pred = lin_reg.predict(X_test)
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Depth']
X = df_c[['IW', 'IF', 'VW', 'FP']]
y = df_c['Depth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
После обучения модели производим предсказания и оцениваем качество модели с помощью показателей MAE и RMSE:
prediction_lr = lr.predict(X_test)
print('LinearRegression_Depth')
print('MAE', mean_absolute_error(y_test, prediction_lr))
print('RMSE', np.sqrt(mean_squared_error(y_test, prediction_lr)))

train_size = int(len(df) * 0.8)
train, test = df_c.iloc[:train_size], df_c.iloc[train_size:]
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Width']
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
X_test = test[['IW', 'IF', 'VW', 'FP']]
y_test = test['Width']
y_pred = lin_reg.predict(X_test)
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Width']
X = df_c[['IW', 'IF', 'VW', 'FP']]
y = df_c['Width']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
После обучения модели производим предсказания и оцениваем качество модели с помощью показателей MAE и RMSE:
prediction_lr = lr.predict(X_test)
print('LinearRegression_Width')
print('MAE', mean_absolute_error(y_test, prediction_lr))
print('RMSE', np.sqrt(mean_squared_error(y_test, prediction_lr)))
![image](https://github.com/user-attachments/assets/ff50d0a2-5a61-44f5-b292-3ea7cd5422f2)
![image](https://github.com/user-attachments/assets/017e24e6-496d-47a0-b400-009a6b7b0bab)


### Оценка модели для DEPTH и WIDTH
Также считается значение MSE и R²:
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')
r_squared = r2_score(y_test, y_pred)
print(f'R²: {r_squared:.2f}')

### Построение модели XGBoost
Данные делятся на обучающую и тестовую выборки:
train_size = int(len(df) * 0.8)
train, test = df_c.iloc[:train_size], df_c.iloc[train_size:]
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Depth']
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
X_test = test[['IW', 'IF', 'VW', 'FP']]
y_test = test['Depth']
y_pred = lin_reg.predict(X_test)
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Depth']
X = df_c[['IW', 'IF', 'VW', 'FP']]
y = df_c['Depth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror')
xg_reg.fit(X_train, y_train)
После обучения модели производим предсказания и оцениваем качество модели с помощью показателей MAE и RMSE:
prediction_xg = xg_reg.predict(X_test)
print('XGBboost_Depth')
print('MAE', mean_absolute_error(y_test, prediction_xg))
print('RMSE', np.sqrt(mean_squared_error(y_test, prediction_xg)))

train_size = int(len(df) * 0.8)
train, test = df_c.iloc[:train_size], df_c.iloc[train_size:]
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Width']
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
X_test = test[['IW', 'IF', 'VW', 'FP']]
y_test = test['Width']
y_pred = lin_reg.predict(X_test)
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Width']
X = df_c[['IW', 'IF', 'VW', 'FP']]
y = df_c['Width']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror')
xg_reg.fit(X_train, y_train)
После обучения модели производим предсказания и оцениваем качество модели с помощью показателей MAE и RMSE:
prediction_xg = xg_reg.predict(X_test)
print('XGBboost_Width')
print('MAE', mean_absolute_error(y_test, prediction_xg))
print('RMSE', np.sqrt(mean_squared_error(y_test, prediction_xg)))

### Оценка модели для DEPTH и WIDTH
Также считается значение MSE и R²:
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')
r_squared = r2_score(y_test, y_pred)
print(f'R²: {r_squared:.2f}')

### Построение модели опорных векторов (SVM) для регрессии
X = df_c[['IW', 'IF', 'VW', 'FP']]
y_depth = df_c['Depth']
y_width = df_c['Width']
X_train, X_test, y_depth_train, y_depth_test = train_test_split(X, y_depth, test_size=0.2, random_state=42)
X_train, X_test, y_width_train, y_width_test = train_test_split(X, y_width, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svr_depth = SVR(kernel='rbf')
svr_depth.fit(X_train, y_depth_train)
svr_width = SVR(kernel='rbf')
svr_width.fit(X_train, y_width_train)
y_depth_pred = svr_depth.predict(X_test)
depth_mse = mean_squared_error(y_depth_test, y_depth_pred)
depth_r2 = r2_score(y_depth_test, y_depth_pred)
y_width_pred = svr_width.predict(X_test)
width_mse = mean_squared_error(y_width_test, y_width_pred)
width_r2 = r2_score(y_width_test, y_width_pred)
![image](https://github.com/user-attachments/assets/e9da1996-a8ed-47e6-9e15-17cd80c64626)
![image](https://github.com/user-attachments/assets/9be7da4a-5468-4eb1-885f-8ee693a8828b)

### Оценка модели для DEPTH и WIDTH
print('SVR results')
print(f'Depth: MSE = {depth_mse}, R² = {depth_r2}')
print(f'Width: MSE = {width_mse}, R² = {width_r2}')

### Делаем выводы по итогам моделирования
Linear Regression и XGBoost показываю poor performance, с отрицательными R² значениями, что говорит о необходимости улучшения или выбора других моделей для предсказания.
SVR показывает значительно лучшие результаты с низкими значениями MSE и высокими R², что делает его наиболее предпочтительным выбором для задач регрессии по переменным Depth и Width. 
Существенно лучшее качество предсказания в SVR предполагает, что в данной задаче метод поддерживающих векторов лучше подходит для сложных зависимостей в данных.
В связи с этим для обучения нейросети в дальнейшем используем модель SVR

### Строим нейросеть
X = df_c[['IW', 'IF', 'VW', 'FP']]
y_depth = df_c['Depth']
y_width = df_c['Width']
Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_depth_train, y_depth_test = train_test_split(X, y_depth, test_size=0.2, random_state=42)
_, _, y_width_train, y_width_test = train_test_split(X, y_width, test_size=0.2, random_state=42)
Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Построение нейросети для предсказания DEPTH
model_depth = Sequential()
model_depth.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))  # Первый слой
model_depth.add(Dense(16, activation='relu'))  # Второй слой
model_depth.add(Dense(1, activation='linear'))  # Выходной слой
Компиляция модели
model_depth.compile(loss='mean_squared_error', optimizer='adam')
Обучение модели
model_depth.fit(X_train, y_depth_train, epochs=100, batch_size=10, verbose=0)
Прогнозирование для DEPTH
y_depth_pred = model_depth.predict(X_test)
Оценка модели для DEPTH
depth_mse = mean_squared_error(y_depth_test, y_depth_pred)
depth_r2 = r2_score(y_depth_test, y_depth_pred)
Построение нейросети для предсказания WIDTH
model_width = Sequential()
model_width.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))  # Первый слой
model_width.add(Dense(16, activation='relu'))  # Второй слой
model_width.add(Dense(1, activation='linear'))  # Выходной слой
Компиляция модели
model_width.compile(loss='mean_squared_error', optimizer='adam')
Обучение модели
model_width.fit(X_train, y_width_train, epochs=100, batch_size=10, verbose=0)
Прогнозирование для WIDTH
y_width_pred = model_width.predict(X_test)
Оценка модели для WIDTH
width_mse = mean_squared_error(y_width_test, y_width_pred)
width_r2 = r2_score(y_width_test, y_width_pred)
Вывод результатов
print(f'Depth: MSE = {depth_mse}, R² = {depth_r2}')
print(f'Width: MSE = {width_mse}, R² = {width_r2}')

### Делаем выводы по полученным результатм от нейросети
DEPTH:  
- Среднеквадратичная ошибка (MSE): 0.012154420321437832  
  Это значение показывает, насколько сильно предсказания модели отклоняются от фактических значений. Чем ниже значение MSE, тем лучше модель.
- Коэффициент детерминации (R²): 0.8061733756394851  
  Это значение указывает на то, какая доля изменения зависимой переменной объясняется независимыми переменными в модели. Значение R² близкое к 1 говорит о хорошей предсказательной способности модели.
WIDTH:  
- Среднеквадратичная ошибка (MSE): 0.012575956899994993  
  Аналогично DEPTH, это значение также низкое, что свидетельствует о неплохих предсказаниях модели.
- Коэффициент детерминации (R²): 0.8186448975954712  
  Указывает на то, что модель хорошо объясняет колебания зависимой переменной, даже чуть лучше, чем в случае с DEPTH.
Таким образом обе модели показывают хорошие результаты с низкими значениями MSE и высокими значениями R², что говорит о том, что на основе данных можно эффективно делать предсказания.


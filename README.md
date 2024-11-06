ОТЧЕТ

В данном отчете представлен процесс анализа и моделирования данных, используя библиотеку Pandas для обработки данных и библиотеки Scikit-learn, Keras и XGBoost для построения моделей. 

### Импорт библиотек
Начинаем с импорта необходимых библиотек:
Python
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
Python
df = pd.read_csv('ebw_data.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.info())
Удаляем дубликаты, если они имеются в данных:
Python
df = df.drop_duplicates()
df_c = df
df_c.to_csv('cleaned_df_final_work.csv')

### Удаление выбросов
Функция remove_outliers применяется для обработки колонок, заданных в списке columns_for_remove_outliers, с целью удаления выбросов на основе межквартильного размаха (IQR):
Python
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
Python
columns_for_tests_and_visualisation = ['IW', 'IF', 'VW', 'FP', 'Depth', 'Width']
for column in columns_for_tests_and_visualisation:
    plt.figure()
    st.probplot(df_c[column], dist='norm', plot=plt)
    plt.title(f'Q-Q график для колонки {column}')
    plt.savefig(f'Q-Q график для колонки {column}.png')
Также создается тепловая карта для визуализации корреляции между переменными:
Python
plt.figure(figsize=(16, 14))
corr_matrix = df_c[columns_for_tests_and_visualisation].corr()
sb.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig('Тепловая карта.png')
![image](https://github.com/user-attachments/assets/27a49278-7c9b-464d-a8ba-957e060a969c)


### Проверка нормальности распределения с помощью теста Андерсона
Проводим тест Андерсона для проверки нормальности распределения для колонок Depth и Width:
Python
result = st.anderson(df_c['Depth'])
# Аналогичные действия для колонки Width

### Построение модели линейной регрессии
Данные делятся на обучающую и тестовую выборки:
Python
train_size = int(len(df) * 0.8)
train, test = df_c.iloc[:train_size], df_c.iloc[train_size:]
X_train = train[['IW', 'IF', 'VW', 'FP']]
y_train = train['Depth']
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
После обучения модели производим предсказания и оцениваем качество модели с помощью показателей MAE и RMSE:
Python
prediction_lr = lin_reg.predict(X_test)
print('MAE', mean_absolute_error(y_test, prediction_lr))
print('RMSE', np.sqrt(mean_squared_error(y_test, prediction_lr)))

### Оценка модели
Также считается значение MSE и R²:
Python
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')
r_squared = r2_score(y_test, y_pred)
print(f'R²: {r_squared:.2f}')

### Визуализация результатов
График предсказаний линейной регрессии по сравнению с фактическими значениями:
Python
plt.subplot(1, 2, 1)
plt.scatter(y_test, prediction_lr, color='blue', label='Предсказания ЛР')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Идеальное предсказание')
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Предсказания Линейной Регрессии')
plt.legend()
plt.show()



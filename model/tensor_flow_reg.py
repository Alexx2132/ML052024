import tensorflow as tf
import numpy as np

# Генерация данных
np.random.seed(0)
data = np.random.rand(100, 3)  # 100 строк, 3 признака

# Целевые значения: первое значение - сумма, второе - квадрат суммы
all_y_trues = np.array([[np.sum(x), np.sum(x**2)] for x in data])

# Создание модели
l0 = tf.keras.layers.Dense(units=4, input_shape=(3,), activation='relu')
l1 = tf.keras.layers.Dense(units=4, activation='relu')
l2 = tf.keras.layers.Dense(units=2, activation='linear')

model = tf.keras.Sequential([l0, l1, l2])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(data, all_y_trues, epochs=500, verbose=False)
print("Закончили обучение модели")

# Пример предсказания
test_input = np.array([[0.2, 0.5, 0.7]])
predicted_output = model.predict(test_input)
print("Предсказанные значения (сумма, квадрат суммы):", predicted_output[0])

# Сохранение модели
model.save('model_sum_squared.h5')

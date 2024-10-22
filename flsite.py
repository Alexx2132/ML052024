import pickle

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron
import pandas as pd
import os
from sklearn.metrics import recall_score, precision_score, accuracy_score




app = Flask(__name__)

menu = [{"name": "kNN", "url": "p_knn"},
        {"name": "Логистическая регрессия", "url": "p_lab2"},
        {"name": "Линейная регрессия", "url": "p_lab3"},
        {"name": "Дерево решений", "url": "p_lab4"},
        {"name": "neuron", "url": "p_lab5"},
        {"name": "Fashion", "url": "p_lab6"}]

loaded_model_knn = pickle.load(open('C:/projects/Project10/model/knn.bin', 'rb'))
new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('C:/projects/Project10/model/neuron_weights.txt')
model_sum_squared = tf.keras.models.load_model('C:/projects/Project10/model/model_sum_squared.h5')
model = load_model('C:/projects/Project10/model/fashion_mnist_model.h5')

class_names = ['Футболка', 'Брюки', 'Пуловер', 'Платье', 'Куртка',
               'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки']
@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Мишиным А.М.", menu=menu)




@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    X_test = pickle.load(open('C:/projects/Project10/model/X_test_knn.pkl', 'rb'))
    Y_test = pickle.load(open('C:/projects/Project10/model/Y_test_knn.pkl', 'rb'))

    if request.method == 'GET':
        return render_template('lab1.html', title="Определение типа аудитории (KNN)", menu=menu, class_model='')

    if request.method == 'POST':
        try:
            tables = float(request.form['tables'])
            chairs = float(request.form['chairs'])
            X_new = np.array([[tables, chairs]])

            pred = loaded_model_knn.predict(X_new)
            type_dict = {0: "Лекционная аудитория", 1: "Практическая аудитория"}
            result = type_dict[pred[0]]


            y_pred = loaded_model_knn.predict(X_test)
            accuracy = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred, average='weighted')
            recall = recall_score(Y_test, y_pred, average='weighted')

            metrics = f"Метрики качества модели: Точность (accuracy) - {accuracy:.2f}, " \
                      f"Точность (precision) - {precision:.2f}, Полнота (recall) - {recall:.2f}"

            return render_template('lab1.html', title="Определение типа аудитории (KNN)", menu=menu,
                                   class_model=f"Это: {result}<br>{metrics}")
        except ValueError as e:
            return render_template('lab1.html', title="Определение типа аудитории (KNN)", menu=menu,
                                   class_model=f"Ошибка: {e}")



@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    with open('C:/projects/Project10/model/logistic_regression_model.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        accuracy = data['accuracy']
        precision = data['precision']
        recall = data['recall']
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model='', accuracy='', precision='', recall='')
    if request.method == 'POST':
        try:
            X_new = np.array([[float(request.form['list1']), float(request.form['list2'])]])
            pred = model.predict(X_new)

            if pred[0] == 1:
                result_label = "лекционная"
            else:
                result_label = "практическая"

            return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                                   class_model="Это " + result_label + " аудитория",
                                   accuracy=f"Accuracy: {accuracy:.2f}",
                                   precision=f"Precision: {precision:.2f}",
                                   recall=f"Recall: {recall:.2f}")
        except ValueError as e:
            return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model='Ошибка в данных', accuracy='', precision='', recall='')





@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    with open('C:/projects/Project10/model/linear_regression_model.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        mse = data['mse']
        mae = data['mae']
        r2 = data['r2']
    if request.method == 'GET':
        return render_template('lab3.html', title="Линейная регрессия", menu=menu, class_model='', mse='', mae='', r2='')
    if request.method == 'POST':
        try:
            X_new = np.array([[float(request.form['height']),
                               float(request.form['weight']),
                               int(request.form['gender'])]])
            pred = model.predict(X_new)
            pred_size = int(round(pred[0]))

            return render_template('lab3.html', title="Линейная регрессия", menu=menu,
                                   class_model=f"Предполагаемый размер обуви: {pred_size}",
                                   mse=f"MSE: {mse:.2f}",
                                   mae=f"MAE: {mae:.2f}",
                                   r2=f"R^2: {r2:.2f}")
        except ValueError as e:
            return render_template('lab3.html', title="Линейная регрессия", menu=menu, class_model='Ошибка в данных', mse='', mae='', r2='')




@app.route("/p_lab4", methods=['POST', 'GET'])
def f_lab4():
    with open('C:/projects/Project10/model/Dtree.bin', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        accuracy = data['accuracy']
        precision = data['precision']
        recall = data['recall']
    if request.method == 'GET':
        return render_template('lab4.html', title="Дерево решений (DecisionTree)", menu=menu, class_model='', accuracy='', precision='', recall='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['desks']),
                           float(request.form['chairs'])]])
        pred = model.predict(X_new)
        class_type = 'лекционная' if pred[0] == 0 else 'практическая'

        return render_template('lab4.html', title="Дерево решений (DecisionTree)", menu=menu,
                               class_model=f"Это {class_type} аудитория",
                               accuracy=f'Accuracy: {accuracy:.2f}',
                               precision=f'Precision: {precision:.2f}',
                               recall=f'Recall: {recall:.2f}')


@app.route("/p_lab5", methods=['POST', 'GET'])
def p_lab5():
    if request.method == 'GET':
        return render_template('neuron.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        try:
            input_features = np.array([[float(request.form['diameter']),
                                        float(request.form['leaf_diameter']),
                                        float(request.form['height'])]])
        except ValueError:
            return render_template('neuron.html', title="Первый нейрон", menu=menu, class_model='Ошибка в данных')

        # Получение предсказаний от модели
        predictions = new_neuron.forward(input_features)
        class_name = np.where(predictions >= 0.5, 'Высшие растения', 'Низшие растения')[0]
        print("Предсказанные значения:", predictions, class_name)

        return render_template('neuron.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + class_name)


@app.route('/p_lab6', methods=['GET', 'POST'])
def p_lab6():
    if request.method == 'GET':
        return render_template('lab6.html', title="Сеть для распознавания одежды", menu=menu)
    if request.method == 'POST':
        # Проверка на наличие файла в запросе
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Сохранение файла на сервере
            filepath = os.path.join('C:/projects/Project10/uploads', file.filename)
            file.save(filepath)

            # Предсказание
            img = image.load_img(filepath, target_size=(28, 28), color_mode="grayscale")
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]

            return render_template('lab6.html', title="Сеть для распознавания одежды", menu=menu,
                                   class_model=predicted_class)


@app.route('/api', methods=['GET'])
def get_classification():
    with open('C:/projects/Project10/model/Dtree.bin', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        accuracy = data['accuracy']
        precision = data['precision']
        recall = data['recall']

    desks = request.args.get('desks')
    chairs = request.args.get('chairs')

    if desks is None or chairs is None:
        return jsonify({'error': 'Не введено количество стульев или парт.'}), 400

    try:
        X_new = np.array([[float(desks), float(chairs)]])
        pred = model.predict(X_new)
        class_type = 'лекционная' if pred[0] == 0 else 'практическая'

        return jsonify({
            'class_model': class_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        })
    except ValueError:
        return jsonify({'error': 'Неверные входные данные. Пожалуйста, укажите числовые значения для столов и стульев.'}), 400


@app.route('/api_reg', methods=['GET'])
def predict_sum_squared():
    try:
        # Получение данных из запроса, три параметра (например, http://localhost:5000/api_sum_squared?p1=0.2&p2=0.5&p3=0.7)
        p1 = float(request.args.get('p1'))
        p2 = float(request.args.get('p2'))
        p3 = float(request.args.get('p3'))

        # Формируем входной массив из трех значений
        input_data = np.array([[p1, p2, p3]])

        # Выполнение предсказания
        predictions = model_sum_squared.predict(input_data)

        # Формирование ответа
        response = {
            'sum': str(predictions[0][0]),
            'sum_squared': str(predictions[0][1])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)


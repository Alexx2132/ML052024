import pickle

import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
from sklearn.metrics import recall_score, precision_score, accuracy_score

app = Flask(__name__)

menu = [{"name": "kNN", "url": "p_knn"},
        {"name": "Логистическая регрессия", "url": "p_lab2"},
        {"name": "Линейная регрессия", "url": "p_lab3"},
        {"name": "Дерево решений", "url": "p_lab4"}]

loaded_model_knn = pickle.load(open('C:/projects/Project10/model/knn.bin', 'rb'))

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


if __name__ == "__main__":
    app.run(debug=True)


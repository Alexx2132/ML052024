import pickle
import cv2
import torch

from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify, flash, redirect
import pandas as pd
import os
from sklearn.metrics import recall_score, precision_score, accuracy_score

SECRET_KEY = 'fejevevevnerjvhe'
app = Flask(__name__)
app.config.from_object(__name__)

menu = [
        {"name": "Классификация", "url": "p_lab6"},
        {"name": "Детектирование", "url": "p_lab7"}]

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/projects/Kursovik/model/best.pt')
model_v = tf.keras.models.load_model('C:/projects/Kursovik/model/furniture_vgg16.h5')


class_names = ['кровать', 'стул', 'диван', 'компьютерное кресло (вращающееся кресло)', 'стол']
detection_class_names = ['Chair', 'Sofa', 'Table']
app.config['UPLOAD_FOLDER'] = 'C:/projects/Kursovik/uploads'
app.config['DETECTION_FOLDER'] = os.path.join('static', 'media', 'detections')


def draw_detections(image_path, detections, save_path):
    img = cv2.imread(image_path)
    for detection in detections:
        # Извлекаем координаты и имя класса
        x_min, y_min = int(detection['x_min']), int(detection['y_min'])
        x_max, y_max = int(detection['x_max']), int(detection['y_max'])
        label = f"{detection['class']} ({detection['confidence'] * 100:.1f}%)"

        # Рисуем прямоугольник на изображении
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Зеленая рамка
        # Добавляем текст
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Синий текст

    # Сохраняем измененное изображение
    cv2.imwrite(save_path, img)

@app.route("/")
def index():
    return render_template('index.html', title='ML-система для решения задач в предметной области "Мебель" ', menu=menu)

@app.route('/p_lab6', methods=['GET', 'POST'])
def p_lab6():
    if request.method == 'GET':
        return render_template('lab6.html', title="Классификация мебели", menu=menu)

    if request.method == 'POST':
        # Проверка наличия файла
        if 'file' not in request.files:
            return "Файл не был загружен"

        file = request.files['file']
        if file.filename == '':
            flash("Вы не выбрали изображение! Выберите файл и повторите попытку.", 'error')
            return redirect(url_for('p_lab6'))

        if file:
            # Сохранение файла
            filepath = os.path.join('C:/projects/Kursovik/uploads', file.filename)
            file.save(filepath)

            try:
                # Обработка изображения
                img = image.load_img(filepath, target_size=(150, 150))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Предсказание
                predictions = model_v.predict(img_array)
                predicted_index = np.argmax(predictions)

                # Преобразование индекса в название класса
                # Используем заранее созданный список class_names
                predicted_class = class_names[predicted_index]

                return render_template('lab6.html',
                                       title="Классификация мебели",
                                       menu=menu,
                                       class_model=predicted_class)
            except Exception as e:
                return f"Ошибка во время распознавания изображения: {e}"

@app.route('/p_lab7', methods=['GET', 'POST'])
def p_lab7():
    if request.method == 'GET':  # Возвращаем HTML форму
        return render_template('lab7.html', title="Детектирование мебели", menu=menu)

    if request.method == 'POST':
        # Проверка на наличие файла в запросе
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            flash("Вы не выбрали изображение для детектирования! Выберите файл и повторите попытку.", 'error')
            return redirect(url_for('p_lab7'))

        if file:
            # Сохранение оригинального изображения
            input_filepath = os.path.join(app.config['DETECTION_FOLDER'], file.filename)
            file.save(input_filepath)

            # Загружаем изображение для анализа
            img = Image.open(input_filepath)

            # Выполняем детектирование
            results = model(img)

            # Получаем данные о всех найденных объектах
            detection_data = results.pandas().xyxy[0]
            detections = []
            for _, row in detection_data.iterrows():
                detections.append({
                    'class': detection_class_names[int(row['class'])],  # Преобразуем номер класса в имя
                    'confidence': row['confidence'],  # Уверенность модели
                    'x_min': row['xmin'],
                    'y_min': row['ymin'],
                    'x_max': row['xmax'],
                    'y_max': row['ymax']
                })

            # Нарисуем рамки и сохраним измененное изображение
            output_filepath = os.path.join(app.config['DETECTION_FOLDER'], f"processed_{file.filename}")
            draw_detections(input_filepath, detections, output_filepath)
            # Отображаем результаты на странице
            return render_template('lab7.html', title="Детектирование мебели", menu=menu,
                                   detections=detections, filename=output_filepath)


@app.route('/api/classify', methods=['POST'])
def api_classify():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не предоставлен'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    try:
        filepath = os.path.join('C:/projects/Kursovik/uploads/temp', file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model_v.predict(img_array)
        predicted_index = np.argmax(predictions)

        predicted_class = class_names[predicted_index]

        os.remove(filepath)

        return jsonify({'class': predicted_class})
    except Exception as e:
        return jsonify({'error': f'Ошибка во время классификации: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)


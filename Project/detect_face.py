import cv2
import numpy as np

from keras.models import load_model

from dataset import first_input

# Загрузка обученной модели по определению эмоций
model_path = "D:/.Labs/VKR/Neural_Emotions/Project/Project_fer2013_Neurons_model_CNN_v4.68-0.78.hdf5"
emotion_data = load_model(model_path, compile=False)
# Присваивание имен классам
labels = {0: 'злость', 1: 'отвращение', 2: 'страх', 3: 'счастье',
           4: 'грусть', 5: 'удивление', 6: 'нейтрально'}

# Импортирование каскада Хаара
haar_data = cv2.CascadeClassifier("D:/.Labs/VKR/Neural_Emotions/haarcascades/haarcascade_frontalface_alt2.xml")

name = "Disgust"
image_path = "D:/.Labs/VKR/Test/FER-2013/testing/photo/" + name + ".jpg"

# Загрузка изображения
image =  cv2.imread(image_path)
# Преобразование в черно-белый формат
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Детектирование лица с помощью импортированных каскадов
detect_image = haar_data.detectMultiScale(image_gray)

# Функция вставки текста на изображение
def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

size = len(detect_image)
if size != 0:
    for (a, b, w, h) in detect_image:
        # Выделяем лицо квадратом
        cv2.rectangle(image, (a, b),
                      (a + h, b + w), # Определение размеров квадрата
                      (0, 0, 255), 5) # Определение цвета и толщины квадрата
        
        # 1
        emotion_image = image_gray[b:b+h, a:a+w]
        # 2
        emotion_image = cv2.resize(emotion_image, (emotion_data.input_shape[1:3]))
        # 3
        emotion_image = first_input(emotion_image, True)
        # 4
        emotion_image = np.expand_dims(emotion_image, 0)
        emotion_image = np.expand_dims(emotion_image, -1)
        # 5
        classif_emotion = np.argmax(emotion_data.predict(emotion_image))
        emotion_text = labels[classif_emotion]
        draw_text((a, b, w, h), image, emotion_text, (0, 0, 255), 0, -15, 2, 2)
# Сохранение изображения
cv2.imwrite("D:/.Labs/VKR/Test/FER-2013/testing/result/" + "detect " + name + ".jpg", image)
# Вывод изображения
cv2.imshow('Detection Face', image)
cv2.waitKey(0)
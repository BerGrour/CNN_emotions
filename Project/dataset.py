import pandas as pd
import numpy as np
import cv2

def first_input(x, v2=True):
    # массив приводится к типу float, уменьшая объем памяти
    x = x.astype('float32')
    # Нормализация значений в диапазон [0: 1]
    x = x / 255.0
    if v2:
        # Дополнительное масштабирование в диапазон [-1: 1]
        x = x - 0.5
        x = x * 2.0
    return x

class DataManager(object):
    def __init__(self, dataset_name='fer2013',
                 dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        if self.dataset_path is not None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'fer2013':
            self.dataset_path = 'D:/.Labs/VKR/Neural_Emotions/Project/icml_face_data.csv'
        else:
            raise Exception(
                    'Incorrect dataset name')

    def get_data(self):
        ground_truth_data = self._load_fer2013()
        return ground_truth_data

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).values  # as_matrix()
        return faces, emotions


def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    else:
        raise Exception('Invalid dataset name')


def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4,
                'surprise': 5, 'neutral': 6}
    else:
        raise Exception('Invalid dataset name')


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

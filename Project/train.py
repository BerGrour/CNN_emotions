import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import to_categorical

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from model import Neurons
from dataset import DataManager
from dataset import split_data
from dataset import first_input

# Параметры
batch_size = 32
epochs = 100
epochs_range = range(epochs)
input_shape = (48, 48, 1)
validation_split = .2

classes = 7
patience = 50
path = 'D:/.Labs/VKR/Neural_Emotions/Project'

# test
data = pd.read_csv('D:/.Labs/VKR/Test/FER-2013/icml_face_data.csv')
data.head()

def data_lead(data):
    # Создаем массив заполненный нулями
    array_image = np.zeros(shape=(len(data), 48, 48))
    # Создаем массив заполненный параметром emotion из csv файла
    label_image = np.array(list(map(int, data['emotion'])))
    
    # Заполняем массив array_image значениями pixels из csv файла
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        array_image[i] = image
    return array_image, label_image

# Применяем созданный класс для распределения данных на 3 этапа
# 1 этап - тренировка (поступают все параметры изображений предназначенных для Training)
train_array_image, train_label_image = data_lead(data[data['Usage']=='Training'])
# 2 этап - валидации (поступают все параметры изображений предназначенных для PrivateTest)
valid_array_image, valid_label_image = data_lead(data[data['Usage']=='PrivateTest'])
# 3 этап - тестирование (поступают все параметры изображений предназначенных для PublicTest)
test_array_image, test_label_image = data_lead(data[data['Usage']=='PublicTest'])

train_images = train_array_image.reshape((train_array_image.shape[0], 48, 48, 1))
train_images = train_images.astype('float32')/255
valid_images = valid_array_image.reshape((valid_array_image.shape[0], 48, 48, 1))
valid_images = valid_images.astype('float32')/255
test_images = test_array_image.reshape((test_array_image.shape[0], 48, 48, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_label_image)
valid_labels = to_categorical(valid_label_image)
test_labels = to_categorical(test_label_image)


data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# параметры модели
model = Neurons(input_shape, classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

dataset_name = 'fer2013'

print('Training dataset:', dataset_name)

# callbacks
log_file_path = path + '_' + dataset_name + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=patience)
trained_models_path = path + dataset_name + '_Neurons_model_CNN_v4'
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', 
                                                save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# загрузка датасета
data_loader = DataManager(dataset_name, image_size=input_shape[:2])
faces, emotions = data_loader.get_data()
faces = first_input(faces)
num_samples, classes = emotions.shape
train_data, val_data = split_data(faces, emotions, validation_split)
train_faces, train_emotions = train_data

# Запуск обучения интегрируемой модели
history = model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=epochs,  callbacks=callbacks,
                        validation_data=val_data)

print('\t')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc, '\n')
#

# Присваивание параметров тестирования
loss = history.history['loss']
acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']

# Построение гистрограмм
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, valid_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, valid_loss, label='Validation Loss')
plt.legend(loc='lower left')
plt.title('Training and Validation Loss')
plt.show()